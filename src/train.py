import tensorflow as tf
import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pymongo import MongoClient
import gridfs
import shutil

def download_data_from_mongo(mongo_uri, db_name, temp_dir):
    """Downloads dataset from MongoDB GridFS to a local temp directory."""
    print(f"Connecting to MongoDB to download dataset...")
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        fs = gridfs.GridFS(db)
        
        # Find all files in GridFS
        # We expect metadata.class to exist
        files = fs.find()
        count = 0
        
        for grid_out in files:
            if hasattr(grid_out, 'metadata') and grid_out.metadata and 'class' in grid_out.metadata:
                class_name = grid_out.metadata['class']
                filename = grid_out.filename
                
                # Create class directory
                class_dir = os.path.join(temp_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                file_path = os.path.join(class_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(grid_out.read())
                count += 1
        
        print(f"Downloaded {count} images from MongoDB to {temp_dir}")
        return count > 0
        
    except Exception as e:
        print(f"Failed to download data from MongoDB: {e}")
        return False

def retrain_model(model_path, data_dir, epochs=5, batch_size=32):
    """
    Retrains (fine-tunes) the model on new data, adapting to new classes if necessary.
    
    Args:
        model_path (str): Path to the existing model.
        data_dir (str): Directory containing the training data (organized by class).
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        
    Returns:
        tf.keras.Model: The retrained model.
        dict: Training history.
    """

    
    # MongoDB Configuration
    mongo_uri = os.getenv("MONGO_URI")
    db_name = "pest_detection_db"
    
    temp_data_dir = None
    
    # If Mongo URI is set, download data to a temp dir
    if mongo_uri:
        temp_data_dir = "temp_mongo_dataset"
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
        os.makedirs(temp_data_dir, exist_ok=True)
        
        if download_data_from_mongo(mongo_uri, db_name, temp_data_dir):
            data_dir = temp_data_dir # Override data_dir to use the downloaded data
        else:
            print("Using local data directory as fallback.")
            
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Detect classes from data directory
    # Filter out hidden files and __MACOSX folder
    class_names = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) 
                          and not d.startswith('.') 
                          and d != '__MACOSX'])
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    
    if num_classes < 2:
        error_msg = f"Retraining requires at least 2 classes, but found {num_classes} ({class_names}). Please upload a ZIP containing all class folders."
        print(error_msg)
        raise ValueError(error_msg)
    
    # Check if model output matches number of classes
    current_output_shape = model.output_shape[-1]
    if current_output_shape != num_classes:
        print(f"Model output shape ({current_output_shape}) does not match number of classes ({num_classes}). Adapting model...")
        
        # Remove the last layer
        # We get the output of the second to last layer
        # We need to ensure we are connecting to the correct tensor
        x = model.layers[-2].output
        
        # Add a new Dense layer with the correct number of classes
        predictions = Dense(num_classes, activation='softmax', name=f'dense_new_{num_classes}')(x)
        
        # Create new model
        # Use model.inputs instead of model.input to handle potential list inputs (though unlikely here)
        model = Model(inputs=model.inputs, outputs=predictions)
        
        # Recompile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Slightly higher LR for new layer adaptation
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model adapted successfully.")
    else:
        print("Model output shape matches number of classes.")
        # Compile model (keep existing optimizer state if possible, or recompile)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # Data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    target_size = (128, 128)
    
    print(f"Loading data from {data_dir}...")
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        subset='validation'
    )
    
    if train_generator.samples == 0:
        print("No training data found.")
        return model, None

    # Setup callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print("Training complete.")
    
    # Save the updated model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save class names to JSON
    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")
    
    # Save class names to MongoDB
    if mongo_uri:
        try:
            print("Saving class names to MongoDB...")
            client = MongoClient(mongo_uri)
            db = client[db_name]
            config_col = db["config"]
            
            config_col.update_one(
                {"_id": "class_names"},
                {"$set": {"names": class_names}},
                upsert=True
            )
            print("Class names synced to MongoDB.")
        except Exception as e:
            print(f"Failed to sync class names to MongoDB: {e}")
    
    # Cleanup temp dir
    if temp_data_dir and os.path.exists(temp_data_dir):
        print("Cleaning up temporary dataset...")
        shutil.rmtree(temp_data_dir)

    return model, history.history

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/mobilenetv2_finetuned_model_exp4.keras"
    DATA_DIR = "data/train" 
    
    if os.path.exists(DATA_DIR):
        retrain_model(MODEL_PATH, DATA_DIR, epochs=1)
    else:
        print(f"Data directory {DATA_DIR} does not exist.")

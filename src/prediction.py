import tensorflow as tf
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_image

import json
from pymongo import MongoClient

# Define default class names
DEFAULT_CLASS_NAMES = ['aphid', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mite', 'mosquito', 'sawfly', 'stem_borer']

def get_class_names(model_dir="models"):
    """
    Attempts to load class names from MongoDB, then a JSON file.
    Returns default list if not found.
    """
    # Try MongoDB first
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        try:
            client = MongoClient(mongo_uri)
            db = client["pest_detection_db"]
            config_col = db["config"]
            doc = config_col.find_one({"_id": "class_names"})
            if doc and "names" in doc:
                return doc["names"]
        except Exception as e:
            print(f"Error loading class names from MongoDB: {e}")

    # Try local JSON file
    class_names_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading class names from {class_names_path}: {e}")
    
    # Fallback to the list we know from the notebook if JSON doesn't exist yet
    return sorted(["ants", "aphids", "bees", "beetle", "bollworm", "catterpillar", "earthworms", "earwig", "grasshopper", "moth", "sawfly", "slug", "snail", "wasp", "weevil"])

def load_trained_model(model_path):
    """
    Loads the trained Keras model.
    
    Args:
        model_path (str): Path to the .h5 or .keras model file.
        
    Returns:
        tf.keras.Model: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_pest(model, image_path, class_names=None):
    """
    Predicts the pest class for a given image.
    
    Args:
        model (tf.keras.Model): The loaded model.
        image_path (str): Path to the image file.
        class_names (list): List of class names corresponding to model outputs.
        
    Returns:
        dict: Dictionary containing the predicted class and confidence score.
    """
    # Always reload class names to ensure we have the latest after retraining
    if class_names is None:
        class_names = get_class_names()
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    if processed_image is None:
        return None
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Apply softmax if the model output is logits (optional, depends on model)
    # scores = tf.nn.softmax(predictions[0]) 
    scores = predictions[0]
    
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(scores)
    
    # Ensure index is within bounds
    if predicted_class_index >= len(class_names):
         return {
            "class": "Unknown",
            "confidence": "0.00%",
            "all_scores": {}
        }

    predicted_class = class_names[predicted_class_index]
    confidence = 100 * np.max(scores)
    
    result = {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "all_scores": {class_names[i]: float(scores[i]) for i in range(len(class_names))}
    }
    
    return result

if __name__ == "__main__":
    # Example usage
    # Replace with actual paths
    MODEL_PATH = "models/mobilenetv2_finetuned_model_exp4.keras" 
    
    # Update class names based on the notebook content
    # From notebook cell 147: ants beetle earthworms grasshopper slug wasp bees catterpillar earwig moth snail weevil
    # From notebook cell 222: class_names = sorted([d for d in os.listdir(data_dir) ...])
    # We should ideally list the exact sorted class names here.
    # Based on the notebook output in cell 147, the folders seem to be:
    # ants, bees, beetle, catterpillar, earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil
    # Let's update the CLASS_NAMES list.
    
    NEW_CLASS_NAMES = sorted(['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'])

    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
        if model:
            while True:
                image_path = input("\nPlease enter the path to the image you want to predict (or 'q' to quit): ").strip()
                
                if image_path.lower() == 'q':
                    break
                
                # Remove quotes if user dragged and dropped file in terminal
                image_path = image_path.replace("'", "").replace('"', "")
                
                if os.path.exists(image_path):
                    print(f"Predicting for: {image_path}...")
                    # Pass the correct class names
                    # Reload class names in case they changed
                    current_class_names = get_class_names()
                    prediction = predict_pest(model, image_path, class_names=current_class_names)
                    print("\nPrediction Result:")
                    print(f"Class: {prediction['class']}")
                    print(f"Confidence: {prediction['confidence']}")
                    # print(f"All Scores: {prediction['all_scores']}") # Uncomment to see all scores
                else:
                    print("Error: Image file not found. Please check the path.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import sys
import zipfile
from typing import List
from pymongo import MongoClient
import gridfs
from mimetypes import guess_type

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction import load_trained_model, predict_pest, get_class_names
from src.train import retrain_model

app = FastAPI(title="Pest Detection API", description="API for detecting agricultural pests from images.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = "models/mobilenetv2_finetuned_model_exp4.keras"
# Directory to store uploaded training data (Persistent Data Lake)
# Directory to store uploaded training data (Persistent Data Lake)
# For local dev, we might still use this, but for Mongo we use GridFS
TRAIN_DATA_DIR = "data/lake"

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "pest_detection_db"

model = None
mongo_client = None
grid_fs = None

@app.on_event("startup")
async def startup_event():
    global model, mongo_client, grid_fs
    # Ensure data lake directory exists (still useful for temp operations)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    
    # Connect to MongoDB
    if MONGO_URI:
        try:
            mongo_client = MongoClient(MONGO_URI)
            db = mongo_client[DB_NAME]
            grid_fs = gridfs.GridFS(db)
            print("Connected to MongoDB Atlas.")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
    else:
        print("Warning: MONGO_URI not set. Running in local mode (no persistent data lake).")
    
    if os.path.exists(MODEL_PATH):
        try:
            model = load_trained_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pest Detection API"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    global prediction_count
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prediction_count += 1
    
    # Save uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Make prediction
        current_class_names = get_class_names()
        print(f"DEBUG: Loaded {len(current_class_names)} class names: {current_class_names}")
        
        prediction = predict_pest(model, temp_file_path, class_names=current_class_names)
        
        if prediction is None:
             raise HTTPException(status_code=400, detail="Failed to process image")
             
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Global training status
training_status = {"status": "idle", "message": ""}
# Global prediction counter
prediction_count = 0

def run_retraining_task():
    """
    Background task to run retraining.
    """
    global model, training_status
    print("Starting background retraining task...")
    training_status = {"status": "training", "message": "Training in progress..."}
    
    try:
        # Ensure training data exists
        if not os.path.exists(TRAIN_DATA_DIR):
            print("No training data found.")
            training_status = {"status": "failed", "message": "No training data found."}
            return

        # Run retraining with sufficient epochs for good performance
        # EarlyStopping callback will stop training if model converges earlier
        new_model, history = retrain_model(MODEL_PATH, TRAIN_DATA_DIR, epochs=20)
        
        # Update global model
        model = new_model
        print("Retraining complete and model updated.")
        training_status = {"status": "completed", "message": "Retraining completed successfully."}
        
    except Exception as e:
        print(f"Retraining failed: {e}")
        training_status = {"status": "failed", "message": f"Retraining failed: {str(e)}"}

@app.get("/retrain/status")
def get_retraining_status():
    return training_status

@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": prediction_count,
        "api_status": "Online",
        "model_version": "v1.0" # Placeholder for versioning
    }

@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a ZIP file containing images organized by class folders to trigger retraining.
    Structure of ZIP:
    root/
      class_a/
        img1.jpg
      class_b/
        img2.jpg
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")
        
    # Save zip file
    zip_path = f"temp_train_data.zip"
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract zip into the Data Lake (merging with existing data)
        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TRAIN_DATA_DIR)
            
        # Handle nested folders (e.g. user zipped a folder 'dataset' containing class folders)
        # Check if TRAIN_DATA_DIR contains only one folder and no images
        items = [item for item in os.listdir(TRAIN_DATA_DIR) if not item.startswith('.') and item != '__MACOSX']
        
        for item in items:
            item_path = os.path.join(TRAIN_DATA_DIR, item)
            if os.path.isdir(item_path):
                # Check if this folder contains sub-folders that look like classes
                sub_items = [s for s in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, s)) and not s.startswith('.') and s != '__MACOSX']
                if sub_items:
                    # It has subfolders. Move them up.
                    print(f"Found nested folder '{item}', moving contents up...")
                    for sub in sub_items:
                        src = os.path.join(item_path, sub)
                        dst = os.path.join(TRAIN_DATA_DIR, sub)
                        if os.path.exists(dst):
                            # Merge: move files from src to dst
                            for file in os.listdir(src):
                                shutil.move(os.path.join(src, file), os.path.join(dst, file))
                            shutil.rmtree(src)
                        else:
                            shutil.move(src, dst)
                    
                    # Remove the now empty (or mostly empty) wrapper folder
                    try:
                        shutil.rmtree(item_path)
                    except:
                        pass
            
            # Upload to MongoDB GridFS
            if grid_fs:
                print("Uploading new data to MongoDB...")
                count = 0
                for root, dirs, files in os.walk(TRAIN_DATA_DIR):
                    for file in files:
                        if file.startswith('.'): continue
                        
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, TRAIN_DATA_DIR)
                        parts = relative_path.split(os.sep)
                        
                        if len(parts) >= 2:
                            class_name = parts[0]
                            filename = os.path.basename(file_path)
                            
                            # Check existence
                            if not grid_fs.exists({"filename": filename, "metadata.class": class_name}):
                                try:
                                    mime_type, _ = guess_type(file_path)
                                    with open(file_path, 'rb') as f:
                                        grid_fs.put(
                                            f,
                                            filename=filename,
                                            content_type=mime_type,
                                            metadata={"class": class_name}
                                        )
                                    count += 1
                                except Exception as e:
                                    print(f"Failed to upload {filename}: {e}")
                print(f"Uploaded {count} new images to MongoDB.")
            else:
                print("MongoDB not connected. Data stored locally only.")
        # Trigger background task
        background_tasks.add_task(run_retraining_task)
        
        return {"message": "Training data uploaded and merged into Data Lake. Retraining started in background."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

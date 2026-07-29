import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import random

def get_mongo_connection():
    """Establishes connection to MongoDB."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None, None
    try:
        client = MongoClient(mongo_uri)
        db = client["pest_detection_db"]
        return client, db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None

def fetch_dataset_stats(sample_size=100):
    """
    Fetches statistics about the dataset.
    
    Args:
        sample_size (int): Number of images to sample for detailed analysis (size/color).
        
    Returns:
        dict: {
            "class_counts": pd.DataFrame (Class, Count),
            "sample_data": pd.DataFrame (Class, Width, Height, R, G, B)
        }
    """
    client, db = get_mongo_connection()
    if not client:
        return None

    fs = gridfs.GridFS(db)
    
    # 1. Class Distribution (Scan all files)
    class_counts = {}
    all_files = []
    
    try:
        # We need to iterate to count. find() returns a cursor.
        # This might be slow for huge datasets, but fine for <10k images.
        for grid_out in fs.find():
            if hasattr(grid_out, 'metadata') and grid_out.metadata and 'class' in grid_out.metadata:
                class_name = grid_out.metadata['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                all_files.append(grid_out._id)
                
    except Exception as e:
        print(f"Error scanning GridFS: {e}")
        return None

    df_class_counts = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    
    # 2. Detailed Analysis (Sampled)
    sample_data = []
    
    # Select random sample
    if len(all_files) > sample_size:
        sampled_ids = random.sample(all_files, sample_size)
    else:
        sampled_ids = all_files
        
    for file_id in sampled_ids:
        try:
            grid_out = fs.get(file_id)
            class_name = grid_out.metadata['class']
            
            # Read image
            img_data = grid_out.read()
            image = Image.open(io.BytesIO(img_data))
            
            # Dimensions
            width, height = image.size
            
            # Color Analysis (Mean RGB)
            # Resize to small size for faster processing
            img_small = image.resize((50, 50))
            img_array = np.array(img_small)
            
            if len(img_array.shape) == 3: # RGB
                mean_color = img_array.mean(axis=(0, 1))
                r, g, b = mean_color[0], mean_color[1], mean_color[2]
            else: # Grayscale
                mean_val = img_array.mean()
                r, g, b = mean_val, mean_val, mean_val
                
            sample_data.append({
                'Class': class_name,
                'Width': width,
                'Height': height,
                'Mean R': r,
                'Mean G': g,
                'Mean B': b
            })
            
        except Exception as e:
            print(f"Error processing file {file_id}: {e}")
            continue
            
    df_sample_data = pd.DataFrame(sample_data)
    
    return {
        "class_counts": df_class_counts,
        "sample_data": df_sample_data
    }

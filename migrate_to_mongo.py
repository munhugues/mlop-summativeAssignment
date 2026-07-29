import os
import sys
from pymongo import MongoClient
import gridfs
from mimetypes import guess_type

# Configuration
# User needs to replace this with their actual connection string
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority")
DB_NAME = "pest_detection_db"
DATA_LAKE_DIR = "data/lake"

def migrate_data():
    print(f"Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        fs = gridfs.GridFS(db)
        print("Connected successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        print("Please check your MONGO_URI environment variable or edit the script.")
        return

    if not os.path.exists(DATA_LAKE_DIR):
        print(f"Data lake directory '{DATA_LAKE_DIR}' not found.")
        return

    print(f"Scanning '{DATA_LAKE_DIR}' for images...")
    
    count = 0
    # Walk through the data lake
    for root, dirs, files in os.walk(DATA_LAKE_DIR):
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            
            # Determine class name from parent folder
            # Structure: data/lake/class_name/image.jpg
            relative_path = os.path.relpath(file_path, DATA_LAKE_DIR)
            parts = relative_path.split(os.sep)
            
            if len(parts) < 2:
                print(f"Skipping {file_path}: Not in a class folder.")
                continue
                
            class_name = parts[0]
            filename = os.path.basename(file_path)
            
            # Check if file already exists in GridFS to avoid duplicates
            if fs.exists({"filename": filename, "metadata.class": class_name}):
                print(f"Skipping {filename} (already exists)")
                continue

            # Upload to GridFS
            try:
                mime_type, _ = guess_type(file_path)
                with open(file_path, 'rb') as f:
                    fs.put(
                        f,
                        filename=filename,
                        content_type=mime_type,
                        metadata={
                            "class": class_name,
                            "original_path": relative_path
                        }
                    )
                print(f"Uploaded: {class_name}/{filename}")
                count += 1
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")

    print(f"\nMigration complete. Uploaded {count} new files.")

if __name__ == "__main__":
    if "mongodb+srv://" not in MONGO_URI and "mongodb://" not in MONGO_URI:
        print("⚠️  WARNING: You have not set a valid MongoDB connection string.")
        print("   Please set the MONGO_URI environment variable or edit this script.")
        print("   Example: export MONGO_URI='mongodb+srv://user:pass@cluster.mongodb.net/...'")
        sys.exit(1)
        
    migrate_data()

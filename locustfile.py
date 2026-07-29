from locust import HttpUser, task, between
import os

class PestDetectionUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def predict(self):
        # Path to a sample image for testing
        # Ensure this file exists in the same directory or provide a valid path
        image_path = "data/moth (3).jpg" 
        
        if not os.path.exists(image_path):
            print(f"Warning: Test image '{image_path}' not found. Skipping request.")
            print(f"Current Working Directory: {os.getcwd()}")
            print(f"Files in current dir: {os.listdir('.')}")
            if os.path.exists('data'):
                 print(f"Files in data dir: {os.listdir('data')}")
            return

        with open(image_path, "rb") as image:
            files = {"file": ("moth (3).jpg", image, "image/jpeg")}
            self.client.post("/predict", files=files)

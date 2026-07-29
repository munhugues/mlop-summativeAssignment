import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Loads and preprocesses an image for the model.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for the image (width, height).
        
    Returns:
        numpy.ndarray: Preprocessed image batch of shape (1, width, height, 3).
    """
    try:
        # Load image
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        
        # Convert to array
        img_array = tf.keras.utils.img_to_array(img)
        
        # Expand dimensions to match model input shape (batch_size, height, width, channels)
        img_array = tf.expand_dims(img_array, 0)
        
        # Rescale pixel values (assuming model was trained with 1./255 rescaling)
        # Adjust this based on your specific model's preprocessing requirements
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

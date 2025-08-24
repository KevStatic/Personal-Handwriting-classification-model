# scripts/predict.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pickle

def preprocess_image(image_path, img_size=28):
    """Loads and preprocesses a single image for prediction."""
    try:
        img = Image.open(image_path).convert('L') # Grayscale
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

def predict_image(model_path, image_path, char_map_path):
    """Loads a model and predicts the class of an image."""
    # Load the model and character map
    try:
        model = keras.models.load_model(model_path)
        with open(char_map_path, 'rb') as f:
            char_map = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model or char map not found. Please run train.py first.")
        return

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None:
        return

    # Make the prediction
    print("Making prediction...")
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    predicted_char = char_map[predicted_class]

    # Display the result
    print(f"\n--- Prediction Result ---")
    print(f"Predicted character: {predicted_char}")
    print(f"Confidence score: {prediction[0][predicted_class]:.4f}")

if __name__ == "__main__":
    model_file = '../saved_models/my_handwriting_model.h5'
    char_map_file = '../saved_models/char_map.pkl'

    # Example usage: Change this to the path of your test image
    test_image_path = '../test_images/sample_C.jpg' 

    if os.path.exists(test_image_path):
        predict_image(model_file, test_image_path, char_map_file)
    else:
        print(f"Please place an image in the 'test_images' folder and update the `test_image_path` variable.")
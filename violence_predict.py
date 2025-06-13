import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

# Configuration
SEQUENCE_LENGTH = 16
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

# Class labels (adjust as needed)
CLASS_NAMES = ["Non-Violence", "Violence"]

# Constants for annotation
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0) # Green

# Load the trained model
try:
    model = load_model("models/last_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Create the results folder if it doesn't exist
os.makedirs("imgs_results", exist_ok=True)

def preprocess_image(img_path):
    """Load and preprocess a single image to generate a sequence."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    # Repeat the same image SEQUENCE_LENGTH times using np.tile
    sequence = np.tile(np.expand_dims(img, axis=0), (SEQUENCE_LENGTH, 1, 1, 1))
    return sequence

def predict_and_annotate(image_path):
    """Predict class and annotate the image with the result."""
    sequence = preprocess_image(image_path)
    if sequence is None:
        return # Exit if image couldn't be preprocessed

    input_batch = np.expand_dims(sequence, axis=0)  # Shape: (1, 16, 64, 64, 3)

    predictions = model.predict(input_batch, verbose=0) # Suppress Keras output
    
    if predictions.shape[1] == 1:
        # Binary classification with sigmoid activation
        predicted_label = CLASS_NAMES[int(predictions[0] > 0.5)]
    else:
        # Multiclass classification
        predicted_label = CLASS_NAMES[np.argmax(predictions[0])]

    # Reload original image to annotate and save
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read original image for annotation from {image_path}")
        return

    annotated_img = cv2.putText(
        original_img,
        predicted_label,
        (10, 30),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    output_path = os.path.join("imgs_results", os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_img)

    print(f"[✓] Image processed. Result saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python violence_predict.py <path_to_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    predict_and_annotate(image_path)
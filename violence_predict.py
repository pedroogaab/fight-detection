import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print("\n -=-=-=-=-=-=-=    Processing   -=-=-=-=-=-=-= \n")



import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys



# --- CONFIGURATION ---
SEQUENCE_LENGTH = 16
IMG_HEIGHT = 64
IMG_WIDTH = 64
CLASS_NAMES = ["Non-Violence", "Violence"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# --- LOAD MODEL ---
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
    
    # Repeat the same image SEQUENCE_LENGTH times
    sequence = np.tile(np.expand_dims(img, axis=0), (SEQUENCE_LENGTH, 1, 1, 1))
    return sequence


def predict_and_annotate(image_path):
    """Predict class and annotate the image with the result."""
    sequence = preprocess_image(image_path)
    if sequence is None:
        return

    # Add batch dimension for model input
    input_batch = np.expand_dims(sequence, axis=0)

    predictions = model.predict(input_batch, verbose=0)

    # **FIX 1: Correctly extract the scalar value from the prediction array**
    # Access the first element of the batch [0] and the first element of the output [0]
    prediction_value = predictions[0][0]
    
    # Determine the class based on the prediction value
    predicted_index = int(prediction_value > 0.5)
    predicted_label = CLASS_NAMES[predicted_index]

    # **FIX 2: Set text color based on the prediction**
    # Green for Non-Violence (BGR: 0, 255, 0), Red for Violence (BGR: 0, 0, 255)
    text_color = (0, 0, 255) if predicted_label == "Violence" else (255, 0, 0)
    
    # Reload original image to annotate and save
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read original image for annotation from {image_path}")
        return

    # Annotate the image with the correct color
    cv2.putText(
        original_img,
        predicted_label,
        (10, 30),
        FONT,
        FONT_SCALE,
        text_color, # Use the dynamic color
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    output_path = os.path.join("imgs_results", os.path.basename(image_path))
    cv2.imwrite(output_path, original_img)

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
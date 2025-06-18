import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("\n -=-=-=-=-=-=-=      Processing      -=-=-=-=-=-=-= \n")


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import argparse # Importa a biblioteca para parsing de argumentos

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
os.makedirs("videos_results", exist_ok=True) # Nova pasta para resultados de vídeo


def preprocess_frame(frame):
    """Preprocess a single frame for the model."""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return img


def predict_frame(sequence):
    """Predict class and confidence for a given sequence."""
    input_batch = np.expand_dims(sequence, axis=0)
    predictions = model.predict(input_batch, verbose=0)
    
    prediction_value = predictions[0][0] # Probabilidade de ser "Violence"
    
    if prediction_value > 0.5:
        predicted_label = "Violence"
        confidence = prediction_value
    else:
        predicted_label = "Non-Violence"
        confidence = 1 - prediction_value
        
    text_color = (0, 0, 255) if predicted_label == "Violence" else (255, 0, 0) # BGR
    text_to_display = f"{predicted_label}: {confidence:.2%}"
    
    return text_to_display, text_color, predicted_label


def process_image(image_path):
    """Process a single image, predict, annotate, and save."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Para imagens, criamos uma sequência repetindo a mesma imagem
    preprocessed_img = preprocess_frame(img)
    sequence = np.tile(np.expand_dims(preprocessed_img, axis=0), (SEQUENCE_LENGTH, 1, 1, 1))
    
    text_to_display, text_color, _ = predict_frame(sequence)

    # Redimensiona a imagem original para visualização
    display_img = cv2.resize(img, (600, 500))
    
    cv2.putText(
        display_img,
        text_to_display,
        (10, 30),
        FONT,
        FONT_SCALE,
        text_color,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    output_path = os.path.join("imgs_results", os.path.basename(image_path))
    cv2.imwrite(output_path, display_img)
    print(f"[✓] Image processed. Result saved to: {output_path}")
    print(f"    -> Prediction: {text_to_display}")


def process_video(video_path):
    """Process a video, predict frame by frame, annotate, and save."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para MP4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = os.path.join("videos_results", "output_" + os.path.basename(video_path))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    frame_buffer = [] # Buffer para armazenar a sequência de frames

    print(f"[⌛] Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        frame_buffer.append(preprocessed_frame)

        # Mantenha o buffer com o tamanho da sequência desejada
        if len(frame_buffer) == SEQUENCE_LENGTH:
            sequence_to_predict = np.array(frame_buffer)
            text_to_display, text_color, predicted_label = predict_frame(sequence_to_predict)

            # Anotar o frame original (não o redimensionado para o modelo)
            cv2.putText(
                frame,
                text_to_display,
                (10, 30),
                FONT,
                FONT_SCALE,
                text_color,
                FONT_THICKNESS,
                cv2.LINE_AA
            )
            out.write(frame)
            
            # Remove o frame mais antigo do buffer para manter o tamanho da sequência
            frame_buffer.pop(0) 
            
        elif len(frame_buffer) < SEQUENCE_LENGTH:
            # Se ainda não temos frames suficientes, simplesmente escrevemos o frame original
            # sem predição, ou podemos tentar repetir o último frame do buffer
            # Para este exemplo, vamos apenas escrever o frame original sem anotação até que
            # o buffer esteja cheio para a primeira predição.
            # Alternativa: Pode-se optar por não escrever até ter a primeira sequência completa
            out.write(frame)
            

    cap.release()
    out.release()
    print(f"[✓] Video processed. Result saved to: {output_filename}")
    print("    -> Note: Prediction starts after the first", SEQUENCE_LENGTH, "frames are buffered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict violence in images or videos.")
    parser.add_argument("--img", type=str, help="Path to an image file.")
    parser.add_argument("--video", type=str, help="Path to a video file.")

    args = parser.parse_args()

    if args.img:
        if not os.path.exists(args.img):
            print(f"Error: Image file not found: {args.img}")
            sys.exit(1)
        process_image(args.img)
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        process_video(args.video)
    else:
        print("Usage: python violence_predict.py --img <path_to_image.jpg>")
        print("Or:    python violence_predict.py --video <path_to_video.mp4>")
        sys.exit(1)
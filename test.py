import sys
import json
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Constants ===
MODEL_PATH = 'coffee_leaf_disease_model.keras'  # Path to your saved model
IMG_HEIGHT = 128
IMG_WIDTH = 128
FALLBACK_CLASS_NAMES = ['brown_eye_spot', 'leaf_miner', 'leaf_rust', 'healthy']

# Default test images folder (used when no CLI arg is supplied)
DEFAULT_TEST_DIR = r'C:\Users\PC\Videos\DATASET-COFFEE\test'


def load_trained_model(path):
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None
    try:
        m = load_model(path)
        return m
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None


# === Function to load and preprocess an image ===
def prepare_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def load_class_labels():
    metrics_path = os.path.join('artifacts', 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                payload = json.load(f)
            labels = payload.get('class_labels')
            if labels and isinstance(labels, list):
                return labels
        except Exception:
            pass
    return FALLBACK_CLASS_NAMES


# === Predict on a single image ===
def predict_image(model, image_path, class_labels):
    if model is None:
        print("No model loaded. Place the trained model file and try again.")
        return

    image = prepare_image(image_path)
    try:
        prediction = model.predict(image)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return

    # Detect model output format safely
    output_shape = getattr(model, 'output_shape', None)
    out_units = None
    if isinstance(output_shape, (list, tuple)):
        out_units = output_shape[-1]
    binary_output = (out_units == 1)

    # Compute class/probability
    if binary_output:
        prob = float(np.asarray(prediction).ravel()[0])
        probs = np.array([1.0 - prob, prob], dtype=float)
    else:
        probs = np.asarray(prediction).ravel()

    predicted_class = int(np.argmax(probs)) if probs.size else 0
    confidence = float(probs[predicted_class]) if probs.size else 0.0
    label = class_labels[predicted_class] if predicted_class < len(class_labels) else str(predicted_class)
    print(f"Predicted: {label} ({confidence*100:.2f}%)")

    # Print all classes, filling missing with 0%
    print("All classes:")
    for i, lbl in enumerate(class_labels):
        val = float(probs[i]) if i < probs.size else 0.0
        print(f"  - {lbl}: {val * 100:.2f}%")


def print_usage():
    print("Usage: python test.py <image_path>")
    print("If no <image_path> is provided the script will prompt for a path,")
    print("or use the first image found under:")
    print(f"  {DEFAULT_TEST_DIR}")


def find_first_image(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                return os.path.join(root, f)
    return None


if __name__ == '__main__':
    # Determine image path: CLI arg takes precedence, otherwise pick first image from DEFAULT_TEST_DIR
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            sys.exit(1)
        if os.path.isdir(image_path):
            picked = find_first_image(image_path)
            if not picked:
                print(f"No image files found in folder: {image_path}")
                sys.exit(1)
            image_path = picked
    else:
        try:
            user_input = input("Paste image path or filename (Enter to auto-pick): ").strip()
        except Exception:
            user_input = ''
        if user_input:
            image_path = user_input
            if not os.path.isabs(image_path):
                # Try relative to current folder
                rel = os.path.abspath(os.path.join(os.getcwd(), image_path))
                if os.path.exists(rel):
                    image_path = rel
            if not os.path.exists(image_path):
                print(f"Test image not found: {image_path}")
                sys.exit(1)
            if os.path.isdir(image_path):
                picked = find_first_image(image_path)
                if not picked:
                    print(f"No image files found in folder: {image_path}")
                    sys.exit(1)
                image_path = picked
        else:
            print(f"No image path provided. Searching for an image under {DEFAULT_TEST_DIR} ...")
            image_path = find_first_image(DEFAULT_TEST_DIR)
            if image_path is None:
                print("No test image found in the default folder.")
                print_usage()
                sys.exit(1)
            print(f"Using: {image_path}")

    model = load_trained_model(MODEL_PATH)
    if model is None:
        print("Model not available. Run `python model.py` to train and save the model, then retry.")
        sys.exit(2)

    class_labels = load_class_labels()
    predict_image(model, image_path, class_labels)

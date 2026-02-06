from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import json
from flask import send_file
import io
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from urllib.parse import urlparse
import time
import shutil
from PIL import Image, ImageDraw, ImageFont

# === Constants ===
MODEL_PATH = 'coffee_leaf_disease_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['brown_eye_spot', 'leaf_miner', 'leaf_rust', 'healthy']
MAX_FILES = 10
DATASET_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
DATASET_MAX_IMAGES = 800
DATASET_MAX_PER_CLASS = 200
AUG_PER_CLASS_DEFAULT = 50

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'DATASET-COFFEE')

# === Load the model (if available) ===
# If the model file is missing, start the app without loading the model so
# the server does not crash. Predictions will return a friendly error.
if not os.path.exists(MODEL_PATH):
    model = None
    binary_output = False
    multi_label = False
    print(f"Warning: model file '{MODEL_PATH}' not found. App started without a model. Run model.py or place the model file and restart to enable predictions.")
else:
    model = load_model(MODEL_PATH)
    # Determine model output format: single sigmoid (1) or softmax (2+)
    output_shape = model.output_shape
    out_units = None
    if isinstance(output_shape, (list, tuple)):
        out_units = output_shape[-1]
    binary_output = (out_units == 1)
    # detect if model is multi-label: multiple outputs with sigmoid activation
    multi_label = False
    try:
        last_activation = getattr(model.layers[-1].activation, '__name__', '')
        if out_units and out_units > 1 and last_activation == 'sigmoid':
            multi_label = True
    except Exception:
        multi_label = False

# Do NOT load validation data or compute metrics here.
# Metrics (confusion matrix, classification report, training history) are generated
# by `model.py` when training finishes and are saved under `artifacts/metrics.json`
# and `static/training_history.png` respectively. The app will read those artifacts
# on demand and will not compute or save metrics at startup.

# Ensure we have a stable class label list for mapping prediction indices to names
class_labels = CLASS_NAMES

# Normalize labels for UI display (e.g., red_spider_mite -> healthy)
def normalize_label(label):
    if not label:
        return label
    raw = str(label).strip()
    key = raw.lower().replace(' ', '_').replace('-', '_')
    if key == 'red_spider_mite':
        return 'healthy'
    return raw

# Friendly display name helper
def display_label_from_raw(raw_label):
    raw = normalize_label(raw_label)
    if not raw:
        return 'UNKNOWN'
    key = str(raw).strip().lower().replace(' ', '_').replace('-', '_')
    label_map = {
        'healthy': 'HEALTHY',
        'leaf_rust': 'LEAF RUST',
        'brown_eye_spot': 'BROWN EYE SPOT',
        'leaf_miner': 'LEAF MINER',
    }
    if key in label_map:
        return label_map[key]
    # default: title-case the class name
    return raw.replace('_', ' ').title()

# If training artifacts exist, prefer the canonical class labels saved by the training script
metrics_path = os.path.join('artifacts', 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as mf:
            payload = json.load(mf)
            class_labels = payload.get('class_labels', class_labels)
    except Exception:
        pass

# Apply label normalization for display
class_labels = [normalize_label(lbl) for lbl in class_labels]

# === Initialize Flask app ===
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Dataset analysis helpers ===
def _detect_class_dirs(root_path):
    try:
        entries = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        entries.sort()
        return entries
    except Exception:
        return []

def _iter_images_in_class(root_path, class_name):
    class_dir = os.path.join(root_path, class_name)
    if not os.path.isdir(class_dir):
        return []
    files = []
    for item in os.listdir(class_dir):
        p = os.path.join(class_dir, item)
        if os.path.isfile(p) and os.path.splitext(item)[1].lower() in DATASET_EXTS:
            files.append(p)
    files.sort()
    return files

def _safe_dataset_path(path_value):
    if not path_value:
        return DEFAULT_DATASET_ROOT
    path_value = path_value.strip().strip('"').strip("'")
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))

def _predict_from_array(img_arr):
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    return format_predictions(prediction, class_labels, threshold=0.5)

def analyze_dataset(dataset_root, max_total=DATASET_MAX_IMAGES, max_per_class=DATASET_MAX_PER_CLASS, augment_per_class=AUG_PER_CLASS_DEFAULT):
    if model is None:
        return {'error': 'Model file not found on server. Train the model or place the model file and restart the app.'}, []
    if not os.path.isdir(dataset_root):
        return {'error': f'Dataset folder not found: {dataset_root}'}, []

    class_dirs = _detect_class_dirs(dataset_root)
    if not class_dirs:
        return {'error': 'No class subfolders found. Expected per-class folders with images.'}, []

    stats = []
    total_seen = 0
    total_correct = 0
    sample_augmented = []

    # Prepare augmentation generator
    aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Output folder for augmented previews
    run_id = str(int(time.time()))
    aug_dir = os.path.join(UPLOAD_FOLDER, 'augmented', run_id)
    os.makedirs(aug_dir, exist_ok=True)

    for class_name in class_dirs:
        class_images = _iter_images_in_class(dataset_root, class_name)
        if not class_images:
            stats.append({'name': class_name, 'total': 0, 'correct': 0, 'accuracy': 0.0})
            continue

        class_total = 0
        class_correct = 0

        # Limit per class and total
        for img_path in class_images[:max_per_class]:
            if total_seen >= max_total:
                break
            try:
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_arr = img_to_array(img)
                formatted = _predict_from_array(img_arr)
                pred_label = ''
                conf = 0.0
                if formatted.get('type') == 'single':
                    pred_label = formatted.get('label', '')
                    conf = float(formatted.get('confidence', 0.0))
                else:
                    preds = formatted.get('predictions', [])
                    if preds:
                        top = max(preds, key=lambda p: p.get('confidence', 0.0))
                        pred_label = top.get('label', '')
                        conf = float(top.get('confidence', 0.0))
                class_total += 1
                total_seen += 1
                if pred_label and pred_label.lower() == class_name.lower():
                    class_correct += 1
                    total_correct += 1
            except Exception:
                # Skip unreadable images
                continue

        acc = (class_correct / class_total) if class_total else 0.0
        stats.append({'name': class_name, 'total': class_total, 'correct': class_correct, 'accuracy': acc})

        # Create a few augmented previews per class (from first image)
        if augment_per_class and class_images:
            try:
                base_img = load_img(class_images[0], target_size=(IMG_HEIGHT, IMG_WIDTH))
                base_arr = img_to_array(base_img)
                for i in range(int(augment_per_class)):
                    aug_arr = aug_gen.random_transform(base_arr)
                    aug_img = Image.fromarray(np.uint8(np.clip(aug_arr, 0, 255)))
                    out_name = secure_filename(f"{class_name}_aug_{i+1}.png")
                    out_path = os.path.join(aug_dir, out_name)
                    aug_img.save(out_path)

                    formatted = _predict_from_array(aug_arr)
                    pred_label = ''
                    conf = 0.0
                    if formatted.get('type') == 'single':
                        pred_label = formatted.get('label', '')
                        conf = float(formatted.get('confidence', 0.0))
                    else:
                        preds = formatted.get('predictions', [])
                        if preds:
                            top = max(preds, key=lambda p: p.get('confidence', 0.0))
                            pred_label = top.get('label', '')
                            conf = float(top.get('confidence', 0.0))

                    sample_augmented.append({
                        'image_url': '/' + out_path.replace('\\', '/'),
                        'true_label': display_label_from_raw(class_name),
                        'pred_label': display_label_from_raw(pred_label),
                        'confidence': conf
                    })
            except Exception:
                pass

    overall_acc = (total_correct / total_seen) if total_seen else 0.0
    dataset_results = {
        'root': dataset_root,
        'total_images': total_seen,
        'overall_accuracy': overall_acc,
        'per_class': stats
    }
    return dataset_results, sample_augmented

# === Function to preprocess image ===
def prepare_image(image):
    # `image` may be a file-like object (uploaded file) or a file path string
    if isinstance(image, str):
        img = load_img(image, target_size=(IMG_HEIGHT, IMG_WIDTH))
    else:
        image_bytes = io.BytesIO(image.read())
        img = load_img(image_bytes, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


# === Helper to format predictions for different model output types ===
def format_predictions(prediction, class_labels, threshold=0.5):
    """Return structured predictions.
    - For binary output: return single predicted class and confidence
    - For softmax multi-class: return single predicted class (argmax) and confidence
    - For multi-label sigmoid: return list of (label, confidence) for probs >= threshold
    """
    try:
        probs = np.array(prediction).ravel()
    except Exception:
        probs = np.array(prediction)

    # Binary single output
    if binary_output:
        prob = float(probs.ravel()[0])
        predicted_class = int(prob > 0.5)
        confidence = prob if predicted_class == 1 else 1.0 - prob
        label = class_labels[predicted_class] if predicted_class < len(class_labels) else str(predicted_class)
        return {'type': 'single', 'predicted_index': predicted_class, 'label': label, 'confidence': float(confidence)}

    # Multi-label: independent sigmoid outputs — return full confidences so UI can filter client-side
    if multi_label:
        results = []
        for idx, p in enumerate(probs):
            lbl = class_labels[idx] if idx < len(class_labels) else str(idx)
            results.append({'index': idx, 'label': lbl, 'confidence': float(p)})
        return {'type': 'multi', 'predictions': results}

    # Otherwise treat as softmax multi-class
    probs = probs.ravel()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    label = class_labels[predicted_class] if predicted_class < len(class_labels) else str(predicted_class)
    return {'type': 'single', 'predicted_index': predicted_class, 'label': label, 'confidence': confidence}


def annotate_image_save(original_path, formatted, out_dir=UPLOAD_FOLDER, suffix='_annotated'):
    """Draw prediction labels onto the image and save an annotated copy in `out_dir`.
    Returns the saved file path or None on failure.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(original_path)
        name, _ext = os.path.splitext(base)
        out_name = secure_filename(f"{name}{suffix}.png")
        out_path = os.path.join(out_dir, out_name)

        img = Image.open(original_path).convert('RGBA')
        w, h = img.size

        # Prepare overlay for semi-transparent background
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Build text lines depending on formatted content
        lines = []
        if formatted is None:
            lines = ['No prediction']
        elif formatted.get('type') == 'single':
            lbl = formatted.get('label', 'Unknown')
            conf = formatted.get('confidence', 0.0)
            lines = [f"{lbl.replace('_',' ').title()}: {conf*100:.1f}%"]
        else:
            preds = formatted.get('predictions', [])
            # sort descending by confidence
            preds = sorted(preds, key=lambda p: p.get('confidence', 0.0), reverse=True)
            for p in preds:
                lines.append(f"{p.get('label','?').replace('_',' ').title()}: {p.get('confidence',0.0)*100:.1f}%")

        # Choose font (fallback to default)
        try:
            font = ImageFont.truetype('arial.ttf', size=16)
        except Exception:
            font = ImageFont.load_default()

        padding = 8
        line_height = font.getsize('Ay')[1] + 4
        box_height = padding * 2 + line_height * len(lines)

        # Draw background rectangle
        rect_width = int(w * 0.6)
        draw.rectangle([0, 0, rect_width, box_height], fill=(0, 0, 0, 150))

        # Draw text lines
        y = padding
        for line in lines:
            draw.text((padding, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_height

        # Composite overlay onto image and save as PNG
        combined = Image.alpha_composite(img, overlay)
        combined.convert('RGB').save(out_path, format='PNG')
        return out_path
    except Exception as e:
        print(f"Failed to annotate image: {e}")
        return None

# === Save classification report as image ===
def save_classification_report_image(report_dict, class_labels, filename='classification_report.png'):
    # keep function in case it's useful elsewhere, but app will not save PNGs here
    metrics = ['precision', 'recall', 'f1-score']
    data = [[report_dict[label][metric] for metric in metrics] for label in class_labels]
    plt.figure(figsize=(8, 4))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=metrics, yticklabels=class_labels)
    plt.title('Classification Report')
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename)
    plt.close()

# Note: confusion matrix and classification report images are NOT saved here.
# The model training script writes numeric metrics to `artifacts/metrics.json` and
# the app serves generated images on-demand from that JSON.

@app.route('/metrics/confusion.png')
def metrics_confusion_png():
    metrics_path = os.path.join('artifacts', 'metrics.json')
    if not os.path.exists(metrics_path):
        return ('', 404)
    with open(metrics_path, 'r') as f:
        payload = json.load(f)
    cm = np.array(payload.get('confusion_matrix', []))
    labels = [normalize_label(lbl) for lbl in payload.get('class_labels', [])]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/metrics/report.png')
def metrics_report_png():
    metrics_path = os.path.join('artifacts', 'metrics.json')
    if not os.path.exists(metrics_path):
        return ('', 404)
    with open(metrics_path, 'r') as f:
        payload = json.load(f)
    class_labels = [normalize_label(lbl) for lbl in payload.get('class_labels', [])]
    classification = payload.get('classification', {})
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    for lbl in class_labels:
        row = [classification.get(lbl, {}).get(m, 0.0) for m in metrics]
        data.append(row)
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=metrics, yticklabels=class_labels, ax=ax)
    ax.set_title('Classification Report')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None
    selected_filename = None
    batch_results = None
    dataset_results = None
    augmented_samples = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded"
        else:
            files = request.files.getlist('file')
            files = [f for f in files if f and f.filename]
            if not files:
                result = "No file selected"
            elif len(files) > 1:
                # Batch mode: classify multiple images
                batch_results = []
                for f in files[:MAX_FILES]:
                    filename = secure_filename(f.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.seek(0)
                    f.save(file_path)
                    image_url = '/' + file_path.replace('\\', '/')

                    if model is None:
                        batch_results.append({
                            'class_label': 'MODEL NOT FOUND',
                            'raw_label': 'model_not_found',
                            'confidence': 0.0,
                            'image_url': image_url,
                        })
                        continue

                    try:
                        with open(file_path, 'rb') as rf:
                            image = prepare_image(rf)
                        prediction = model.predict(image)
                    except Exception as e:
                        batch_results.append({
                            'class_label': 'PREDICTION FAILED',
                            'raw_label': 'prediction_failed',
                            'confidence': 0.0,
                            'image_url': image_url,
                            'message': str(e),
                        })
                        continue

                    formatted = format_predictions(prediction, class_labels, threshold=0.5)
                    if formatted.get('type') == 'single':
                        raw_label = formatted.get('label', '')
                        display_label = display_label_from_raw(raw_label)
                        batch_results.append({
                            'class_label': display_label,
                            'raw_label': raw_label,
                            'confidence': float(formatted.get('confidence', 0.0)),
                            'image_url': image_url,
                        })
                    else:
                        preds = formatted.get('predictions', [])
                        if preds:
                            top = max(preds, key=lambda p: p.get('confidence', 0.0))
                            raw_label = top.get('label', '')
                            display_label = display_label_from_raw(raw_label)
                            conf = float(top.get('confidence', 0.0))
                        else:
                            raw_label = ''
                            display_label = 'UNKNOWN'
                            conf = 0.0
                        batch_results.append({
                            'class_label': display_label,
                            'raw_label': raw_label,
                            'confidence': conf,
                            'image_url': image_url,
                        })

                selected_filename = None
                return render_template('index.html',
                                       batch_results=batch_results,
                                       result=None,
                                       image_url=None,
                                       selected_filename=selected_filename,
                                       report=None,
                                       cm=None,
                                       labels=class_labels,
                                       macro_precision=None,
                                       macro_recall=None,
                                       macro_f1=None,
                                       has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                       has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                       dataset_results=dataset_results,
                                       augmented_samples=augmented_samples,
                                       zip=zip)
            else:
                file = files[0]
                # Save uploaded image
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.seek(0)
                file.save(file_path)
                selected_filename = filename
                image_url = '/' + file_path.replace('\\', '/')

                # Prepare image for prediction
                with open(file_path, 'rb') as f:
                    image = prepare_image(f)
                # NOTE: normal flow uses uploaded file. If no uploaded file, we will look for a pasted path below.
                # If model is not loaded, return a friendly message without attempting prediction
                if model is None:
                    result = {
                        'class_label': 'MODEL NOT FOUND',
                        'confidence': 0.0,
                        'message': 'Model file not found on server. Run model.py to train the model or place the trained model file and restart the app.'
                    }
                    # early render: skip prediction and mapping
                    return render_template('index.html',
                                           result=result,
                                           image_url=image_url,
                                           selected_filename=selected_filename,
                                           report=None,
                                           cm=None,
                                           labels=class_labels,
                                           macro_precision=None,
                                           macro_recall=None,
                                           macro_f1=None,
                                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                           dataset_results=dataset_results,
                                           augmented_samples=augmented_samples,
                                           zip=zip)

                # Model present: compute prediction and format it uniformly
                prediction = model.predict(image)
                formatted = format_predictions(prediction, class_labels, threshold=0.5)

                # Create annotated image and use that in UI when available
                try:
                    annotated = annotate_image_save(file_path, formatted)
                    if annotated:
                        image_url = '/' + annotated.replace('\\', '/')
                except Exception:
                    pass

                if formatted.get('type') == 'single':
                    # Display the predicted class name consistently with the class list
                    raw_label = formatted.get('label', '')
                    display_label = display_label_from_raw(raw_label)
                    result = {
                        'class_label': display_label,
                        'raw_label': raw_label,
                        'confidence': float(formatted.get('confidence', 0.0)),
                    }
                

                else:
                    # multi-label: return list of label/confidence pairs
                    preds = formatted.get('predictions', [])
                    # keep raw labels and confidences for template
                    result = {
                        'predictions': [{'label': p['label'], 'confidence': float(p['confidence'])} for p in preds]
                    }
                return render_template('index.html',
                                       result=result,
                                       image_url=image_url,
                                       selected_filename=selected_filename,
                                       report=None,
                                       cm=None,
                                       labels=class_labels,
                                       macro_precision=None,
                                       macro_recall=None,
                                       macro_f1=None,
                                       has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                       has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                       dataset_results=dataset_results,
                                       augmented_samples=augmented_samples,
                                       zip=zip)

            # If we reach here it means no file was uploaded or file was empty. Check for pasted image path in form data
            image_path_field = request.form.get('image_path', '').strip()
            if image_path_field:
                # If it's an HTTP/HTTPS URL, download it into uploads
                parsed = urlparse(image_path_field)
                try:
                    if parsed.scheme in ('http', 'https'):
                        # derive filename
                        base = os.path.basename(parsed.path) or 'downloaded_image'
                        local_name = secure_filename(base)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], local_name)
                        urllib.request.urlretrieve(image_path_field, file_path)
                        image_url = '/' + file_path.replace('\\', '/')
                        image = file_path
                    else:
                        # Treat as local path or filename relative to uploads
                        if os.path.isabs(image_path_field) and os.path.exists(image_path_field):
                            file_path = image_path_field
                            image_url = '/' + file_path.replace('\\', '/')
                            image = file_path
                        else:
                            # check uploads folder
                            maybe = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_path_field))
                            if os.path.exists(maybe):
                                file_path = maybe
                                image_url = '/' + file_path.replace('\\', '/')
                                image = file_path
                            else:
                                result = 'Could not find image at provided path/URL.'
                                return render_template('index.html', result=result, image_url=None, selected_filename=None,
                                                       report=None, cm=None, labels=class_labels,
                                                       macro_precision=None, macro_recall=None, macro_f1=None,
                                                       has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                                       has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                                       dataset_results=dataset_results,
                                                       augmented_samples=augmented_samples,
                                                       zip=zip)
                except Exception as e:
                    result = f'Failed to retrieve image: {e}'
                    return render_template('index.html', result=result, image_url=None, selected_filename=None,
                                           report=None, cm=None, labels=class_labels,
                                           macro_precision=None, macro_recall=None, macro_f1=None,
                                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                           dataset_results=dataset_results,
                                           augmented_samples=augmented_samples,
                                           zip=zip)

                # Now we have `image` as a file path string; prepare and predict
                if model is None:
                    result = {
                        'class_label': 'MODEL NOT FOUND',
                        'confidence': 0.0,
                        'message': 'Model file not found on server. Run model.py to train the model or place the trained model file and restart the app.'
                    }
                    return render_template('index.html', result=result, image_url=image_url, selected_filename=None,
                                           report=None, cm=None, labels=class_labels,
                                           macro_precision=None, macro_recall=None, macro_f1=None,
                                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                           dataset_results=dataset_results,
                                           augmented_samples=augmented_samples,
                                           zip=zip)

                try:
                    img_arr = prepare_image(image)
                    prediction = model.predict(img_arr)
                except Exception as e:
                    result = f'Model prediction failed: {e}'
                    return render_template('index.html', result=result, image_url=image_url, selected_filename=None,
                                           report=None, cm=None, labels=class_labels,
                                           macro_precision=None, macro_recall=None, macro_f1=None,
                                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                           dataset_results=dataset_results,
                                           augmented_samples=augmented_samples,
                                           zip=zip)

                formatted = format_predictions(prediction, class_labels, threshold=0.5)
                # Annotate downloaded / provided image and prefer annotated version in UI
                try:
                    annotated = annotate_image_save(file_path, formatted)
                    if annotated:
                        image_url = '/' + annotated.replace('\\', '/')
                except Exception:
                    pass
                if formatted.get('type') == 'single':
                    raw_label = formatted.get('label', '')
                    display_label = display_label_from_raw(raw_label)
                    result = {
                        'class_label': display_label,
                        'raw_label': raw_label,
                        'confidence': float(formatted.get('confidence', 0.0)),
                    }
                else:
                    preds = formatted.get('predictions', [])
                    result = {
                        'predictions': [{'label': p['label'], 'confidence': float(p['confidence'])} for p in preds]
                    }

                return render_template('index.html', result=result, image_url=image_url, selected_filename=None,
                                       report=None, cm=None, labels=class_labels,
                                       macro_precision=None, macro_recall=None, macro_f1=None,
                                       has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                                       has_report=os.path.exists(os.path.join('static','classification_report.png')),
                                       dataset_results=dataset_results,
                                       augmented_samples=augmented_samples,
                                       zip=zip)
    return render_template('index.html',
                           result=result,
                           image_url=image_url,
                           selected_filename=selected_filename,
                           batch_results=batch_results,
                           report=None,
                           cm=None,
                           labels=class_labels,
                           macro_precision=None,
                           macro_recall=None,
                           macro_f1=None,
                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                           dataset_results=dataset_results,
                           augmented_samples=augmented_samples,
                           zip=zip  # Pass zip to Jinja2
                           )

@app.route('/dataset', methods=['POST'])
def dataset_analyze():
    dataset_path = request.form.get('dataset_path', '').strip()
    max_total = request.form.get('max_total', '').strip()
    max_per_class = request.form.get('max_per_class', '').strip()
    augment_per_class = request.form.get('augment_per_class', '').strip()

    try:
        max_total_val = int(max_total) if max_total else DATASET_MAX_IMAGES
    except Exception:
        max_total_val = DATASET_MAX_IMAGES
    try:
        max_per_class_val = int(max_per_class) if max_per_class else DATASET_MAX_PER_CLASS
    except Exception:
        max_per_class_val = DATASET_MAX_PER_CLASS
    try:
        augment_per_class_val = int(augment_per_class) if augment_per_class else AUG_PER_CLASS_DEFAULT
    except Exception:
        augment_per_class_val = AUG_PER_CLASS_DEFAULT

    max_total_val = max(1, min(max_total_val, 2000))
    max_per_class_val = max(1, min(max_per_class_val, 2000))
    augment_per_class_val = max(0, min(augment_per_class_val, 12))

    dataset_root = _safe_dataset_path(dataset_path)
    dataset_results, augmented_samples = analyze_dataset(
        dataset_root,
        max_total=max_total_val,
        max_per_class=max_per_class_val,
        augment_per_class=augment_per_class_val
    )

    return render_template('index.html',
                           result=None,
                           image_url=None,
                           selected_filename=None,
                           batch_results=None,
                           report=None,
                           cm=None,
                           labels=class_labels,
                           macro_precision=None,
                           macro_recall=None,
                           macro_f1=None,
                           has_confusion=os.path.exists(os.path.join('static','confusion_matrix.png')),
                           has_report=os.path.exists(os.path.join('static','classification_report.png')),
                           dataset_results=dataset_results,
                           augmented_samples=augmented_samples,
                           zip=zip)

if __name__ == '__main__':
    app.run(debug=True)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib
# Use Agg backend for headless environments (safe for servers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('static', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

IMG_HEIGHT = 128
IMG_WIDTH = 128
VIS_HEIGHT = 128
VIS_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = int(os.environ.get('EPOCHS', '20'))

# Dataset paths: prefer environment variables, otherwise use `DATASET-COFFEE` inside project root.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'DATASET-COFFEE')
TRAIN_DATA_PATH = os.environ.get('TRAIN_DATA_PATH', os.path.join(DEFAULT_DATASET_ROOT, 'train'))
VAL_DATA_PATH = os.environ.get('VAL_DATA_PATH', os.path.join(DEFAULT_DATASET_ROOT, 'val'))

def _detect_classes(path):
    try:
        entries = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        entries.sort()
        return entries
    except Exception:
        return []

def _save_dataset_visualization(dataset_root, class_names, out_path, per_class=4):
    """Save a simple grid of input images for documentation."""
    try:
        rows = max(1, len(class_names))
        cols = max(1, per_class)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        if rows == 1:
            axes = np.array([axes])
        if cols == 1:
            axes = axes.reshape(rows, 1)

        for r, cls in enumerate(class_names):
            cls_dir = os.path.join(dataset_root, cls)
            if not os.path.isdir(cls_dir):
                for c in range(cols):
                    axes[r, c].axis('off')
                continue
            imgs = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
            imgs.sort()
            for c in range(cols):
                ax = axes[r, c]
                ax.axis('off')
                if c < len(imgs):
                    img_path = os.path.join(cls_dir, imgs[c])
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(VIS_HEIGHT, VIS_WIDTH))
                        ax.imshow(img)
                        if c == 0:
                            ax.set_title(cls.replace('_', ' ').title(), fontsize=9)
                    except Exception:
                        pass
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass

def _save_dataset_augmentation_visualization(dataset_root, class_names, out_path, per_class=3):
    """Save a grid of augmented samples for documentation."""
    try:
        rows = max(1, len(class_names))
        cols = max(1, per_class)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        if rows == 1:
            axes = np.array([axes])
        if cols == 1:
            axes = axes.reshape(rows, 1)

        aug_gen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True
        )

        for r, cls in enumerate(class_names):
            cls_dir = os.path.join(dataset_root, cls)
            if not os.path.isdir(cls_dir):
                for c in range(cols):
                    axes[r, c].axis('off')
                continue
            imgs = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
            imgs.sort()
            if not imgs:
                for c in range(cols):
                    axes[r, c].axis('off')
                continue
            base_path = os.path.join(cls_dir, imgs[0])
            try:
                base_img = tf.keras.preprocessing.image.load_img(base_path, target_size=(VIS_HEIGHT, VIS_WIDTH))
                base_arr = tf.keras.preprocessing.image.img_to_array(base_img)
            except Exception:
                base_arr = None
            for c in range(cols):
                ax = axes[r, c]
                ax.axis('off')
                if base_arr is None:
                    continue
                try:
                    aug_arr = aug_gen.random_transform(base_arr)
                    aug_img = tf.keras.preprocessing.image.array_to_img(aug_arr)
                    ax.imshow(aug_img)
                    if c == 0:
                        ax.set_title(cls.replace('_', ' ').title(), fontsize=9)
                except Exception:
                    pass
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass


# Folder-based loader only: expects per-class subfolders with JPG images
train_classes = _detect_classes(TRAIN_DATA_PATH)
NUM_CLASSES = len(train_classes) if train_classes else None
if NUM_CLASSES is None:
    raise RuntimeError(f"Could not find class folders under {TRAIN_DATA_PATH}. Place images in per-class folders or set TRAIN_DATA_PATH/VAL_DATA_PATH.")

# Save a documentation-ready visualization of input images (150x150x3)
vis_out = os.path.join('artifacts', 'dataset_visualization.png')
_save_dataset_visualization(TRAIN_DATA_PATH, train_classes, vis_out, per_class=4)

# Save augmented samples visualization (processing stage)
aug_out = os.path.join('artifacts', 'dataset_augmentation.png')
_save_dataset_augmentation_visualization(TRAIN_DATA_PATH, train_classes, aug_out, per_class=3)

# Choose class_mode automatically: binary for 2 classes, categorical otherwise
if NUM_CLASSES == 2:
    class_mode = 'binary'
else:
    class_mode = 'categorical'

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    shuffle=True
)
val_data = val_datagen.flow_from_directory(
    VAL_DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    shuffle=False
)
# class_labels from generator
class_labels = [k for k, v in sorted(train_data.class_indices.items(), key=lambda x: x[1])]

# === Save fitting parameters table for paper ===
train_samples = getattr(train_data, 'samples', None)
val_samples = getattr(val_data, 'samples', None)
train_shape = (train_samples, IMG_HEIGHT, IMG_WIDTH, 3) if train_samples is not None else 'unknown'
val_shape = (val_samples, IMG_HEIGHT, IMG_WIDTH, 3) if val_samples is not None else 'unknown'

fit_table_path = os.path.join('artifacts', 'fitting_parameters.txt')
with open(fit_table_path, 'w') as f:
    f.write('Parameters\tInput Value\n')
    f.write(f'X_train\t{train_shape}\n')
    f.write(f'X_val\t{val_shape}\n')
    f.write(f'Batch_size\t{BATCH_SIZE}\n')
    f.write(f'Epochs\t{EPOCHS}\n')

fit_csv_path = os.path.join('artifacts', 'fitting_parameters.csv')
with open(fit_csv_path, 'w', encoding='utf-8') as f:
    f.write('parameter,input_value\n')
    f.write(f'X_train,"{train_shape}"\n')
    f.write(f'X_val,"{val_shape}"\n')
    f.write(f'Batch_size,{BATCH_SIZE}\n')
    f.write(f'Epochs,{EPOCHS}\n')

# === Build the MobileNetV2 model ===
try:
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
except Exception:
    # Fallback if ImageNet weights can't be downloaded
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights=None
    )
base_model.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
])

if NUM_CLASSES == 2:
    model.add(Dense(1, activation='sigmoid'))
else:
    model.add(Dense(NUM_CLASSES, activation='softmax'))

# === Compile the model ===
# Compile with appropriate loss for binary vs multiclass/multilabel
if NUM_CLASSES == 2:
    loss = 'binary_crossentropy'
else:
    loss = 'categorical_crossentropy'

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss,
    metrics=['accuracy']
)

# === Save model summary for paper ===
summary_path = os.path.join('artifacts', 'model_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Also save a simple CSV-style layer table (name, type, output_shape, params)
layers_path = os.path.join('artifacts', 'model_layers.csv')
with open(layers_path, 'w', encoding='utf-8') as f:
    f.write('name,type,output_shape,params\n')
    for layer in model.layers:
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = 'unknown'
        try:
            params = layer.count_params()
        except Exception:
            params = 0
        f.write(f"{layer.name},{layer.__class__.__name__},{out_shape},{params}\n")

# === Train the model ===
if isinstance(train_data, tf.data.Dataset):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )
else:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

# === Plot training & validation accuracy/loss ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
# Save training history figure to static so the app (optionally) can display it
plt.savefig(os.path.join('static', 'model_metrics.png'))
# Do not call plt.show() here; this script may run headless in some environments
plt.close()

# === Save the model ===
model.save("coffee_leaf_disease_model.keras")

# === Evaluation metrics ===
# generator evaluation
Y_true = val_data.classes
Y_pred_probs = model.predict(val_data)
if NUM_CLASSES == 2:
    Y_pred = (Y_pred_probs > 0.5).astype(int).ravel()
else:
    Y_pred = np.argmax(Y_pred_probs, axis=1)
class_labels = [k for k, v in sorted(val_data.class_indices.items(), key=lambda x: x[1])]

print("\nClassification Report:")
report_dict = classification_report(Y_true, Y_pred, target_names=class_labels, output_dict=True)
print(classification_report(Y_true, Y_pred, target_names=class_labels))

# Save metrics JSON (not PNG) so the app can generate figures on-demand
metrics = ['precision', 'recall', 'f1-score']
# Non-multilabel case: compute confusion matrix and save images
cm = confusion_matrix(Y_true, Y_pred)
print("\nConfusion Matrix:")
print(cm)

metrics_payload = {
    'class_labels': class_labels,
    'confusion_matrix': cm.tolist(),
    'classification': {label: {m: float(report_dict[label][m]) for m in metrics} for label in class_labels}
}
import json
with open(os.path.join('artifacts', 'metrics.json'), 'w') as f:
    json.dump(metrics_payload, f)

# === Save confusion matrix and classification report as PNG images under static/ ===
conf_png = os.path.join('static', 'confusion_matrix.png')
report_png = os.path.join('static', 'classification_report.png')

# Confusion matrix image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(conf_png)
plt.close()

# Classification report image
plt.figure(figsize=(8, 4))
data = [[report_dict[label][metric] for metric in metrics] for label in class_labels]
sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=metrics, yticklabels=class_labels)
plt.title('Classification Report')
plt.tight_layout()
plt.savefig(report_png)
plt.close()

# === Macro metrics ===
precision = precision_score(Y_true, Y_pred, average='macro')
recall = recall_score(Y_true, Y_pred, average='macro')
f1 = f1_score(Y_true, Y_pred, average='macro')

print(f"\nMacro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1-score:  {f1:.4f}")

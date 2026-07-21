import os
import sys

# SILENCE C++ ENGINE LOGS & ONEDNN NOTICES
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

# DYNAMICALLY FILTER THE WINDOWS GPU / WSL2 WARNING FROM STDERR
class WarningFilterStream:
    def __init__(self, target_stream):
        self.target_stream = target_stream
    def write(self, message):
        if "Please use WSL2" in message or "TensorFlow-DirectML plugin" in message:
            return
        if "tf.placeholder is deprecated" in message:
            return
        self.target_stream.write(message)
    def flush(self):
        self.target_stream.flush()

sys.stderr = WarningFilterStream(sys.stderr)

# SILENCE PYTHON-LEVEL WARNINGS
import warnings #noqa
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# NOW SAFELY IMPORT HEAVY LIBRARIES
import numpy as np #noqa
import tensorflow as tf #noqa
from tensorflow.keras.preprocessing import image #noqa
from tqdm import tqdm #noqa

# Setup directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_sign_model.keras")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset")

def get_dynamic_class_names():
    """Automatically scans the dataset folder to get sorted class names."""

    if not os.path.exists(DATASET_PATH):
        return []
    items = os.listdir(DATASET_PATH)
    classes = [d for d in items if os.path.isdir(os.path.join(DATASET_PATH, d))]
    return sorted(classes)

def evaluate_all_images():
    """
    Run batch inference across all target 
    images and return classification metrics.
    """
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    class_names = get_dynamic_class_names()
    if not class_names:
        print("❌ Error: Could not determine class names from dataset directory.")
        return

    print("🤖 Loading trained RoadEye brain...")
    custom_dict = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_dict, 
        compile=False,
        safe_mode=False
    )

    print("📂 Scanning dataset subfolders for images with TTA Enabled...\n")
    
    # Pre-count total files so tqdm can render a 0% -> 100% progress bar
    image_tasks = []
    for true_class in class_names:
        class_folder_path = os.path.join(DATASET_PATH, true_class)
        for root, _, files in os.walk(class_folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_tasks.append((true_class, os.path.join(root, file)))

    total_images = 0
    correct_predictions = 0
    class_stats = {c: {"total": 0, "correct": 0} for c in class_names}

    try:
        left_idx = class_names.index("turn_left")
        right_idx = class_names.index("turn_right")
    except ValueError:
        left_idx, right_idx = -1, -1

    # Processing loop wrapped in a real-time progress bar
    for true_class, image_path in tqdm(image_tasks, desc="Evaluating RoadEye", unit="img"):
        # 1. Base Image Processing
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Pass 1: Original Clean Prediction
        pred_orig = model.predict(img_array, verbose=0)[0]
        
        # Pass 2: Brightness TTA
        img_bright = np.clip(img_array * 1.15, 0, 255)
        pred_bright = model.predict(img_bright, verbose=0)[0]
        
        # Pass 3: Horizontal Flip TTA
        img_flipped = np.flip(img_array, axis=2)
        pred_flipped = model.predict(img_flipped, verbose=0)[0]
        
        # NumPy-safe temporary variable swap for flipped images
        if left_idx != -1 and right_idx != -1:
            temp_val = pred_flipped[left_idx]
            pred_flipped[left_idx] = pred_flipped[right_idx]
            pred_flipped[right_idx] = temp_val
        
        # Combine predictions via weighted ensemble consensus
        final_predictions = (pred_orig * 0.50) + (pred_bright * 0.25) + (pred_flipped * 0.25)
        highest_probability_idx = np.argmax(final_predictions)
        predicted_class = class_names[highest_probability_idx]
        
        # Update stats
        total_images += 1
        class_stats[true_class]["total"] += 1
        
        if predicted_class == true_class:
            correct_predictions += 1
            class_stats[true_class]["correct"] += 1

    # --- PRINT ACCURACY BREAKDOWN REPORT ---
    print("\n📊 --- ROADEYE EVALUATION REPORT (WITH TTA) --- 📊\n")
    print(f"{'Traffic Sign Class':<25} | {'Correct / Total':<15} | {'Accuracy':<10}")
    print("-" * 58)
    
    for c in class_names:
        total = class_stats[c]["total"]
        correct = class_stats[c]["correct"]
        acc_str = f"{(correct / total * 100):.1f}%" if total > 0 else "0.0%"
        print(f"{c:<25} | {f'{correct}/{total}':<15} | {acc_str:<10}")
        
    print("-" * 58)
    final_score = (correct_predictions / total_images * 100) if total_images > 0 else 0
    print(f"🚀 FINAL OVERALL SCORE: {correct_predictions}/{total_images} ({final_score:.2f}% Match Rate)")

if __name__ == "__main__":
    evaluate_all_images()
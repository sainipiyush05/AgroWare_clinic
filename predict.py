import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Fix for Keras 3 model loading errors
import json
import argparse
import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "agroware_model.h5"
DEFAULT_TFLITE_PATH = "agroware_model.tflite"
DEFAULT_LABELS_PATH = "class_labels.json"
IMG_SIZE = 224
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# Helper Functions
# =============================================================================

def load_and_preprocess_image(image_path):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array


def load_labels(labels_path):
    """Load labels safely"""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found.\nExpected format:\n"
            '{"class_names": ["Class1", "Class2"]}'
        )

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    if 'class_names' not in labels:
        raise ValueError("Labels file must contain 'class_names'")

    return labels


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_keras(image_path, model_path, labels_path, crop_filter=None):
    """Prediction using TensorFlow Keras ONLY (fixed)"""
    import tensorflow as tf

    print(f"📦 Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    labels = load_labels(labels_path)
    class_names = labels['class_names']

    img = load_and_preprocess_image(image_path)

    predictions = model.predict(img, verbose=0)
    
    num_classes = predictions.shape[-1]
    if len(class_names) < num_classes:
        print(f"⚠️ Warning: Model outputs {num_classes} classes but labels file has {len(class_names)}. Pad with defaults.")
        for i in range(len(class_names), num_classes):
            class_names.append(f"Class_{i}")

    # Get Top-K
    top_indices = predictions[0].argsort()[::-1]

    results = []
    for i in top_indices:
        class_name = class_names[i]
        
        # Apply crop filter if provided
        if crop_filter and crop_filter.lower() not in class_name.lower():
            continue
            
        results.append({
            "class": class_name,
            "confidence": float(predictions[0][i])
        })
        
        if len(results) >= TOP_K:
            break

    if not results and crop_filter:
        print(f"⚠️ No matching predictions found for crop '{crop_filter}'.")

    return results


def predict_tflite(image_path, tflite_path, labels_path, crop_filter=None):
    """Prediction using TFLite"""
    import tensorflow as tf

    print(f"📦 Loading TFLite model: {tflite_path}")

    labels = load_labels(labels_path)
    class_names = labels['class_names']

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_and_preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])

    num_classes = predictions.shape[-1]
    if len(class_names) < num_classes:
        for i in range(len(class_names), num_classes):
            class_names.append(f"Class_{i}")

    top_indices = predictions[0].argsort()[::-1]

    results = []
    for i in top_indices:
        class_name = class_names[i]
        
        # Apply crop filter if provided
        if crop_filter and crop_filter.lower() not in class_name.lower():
            continue
            
        results.append({
            "class": class_name,
            "confidence": float(predictions[0][i])
        })
        
        if len(results) >= TOP_K:
            break

    return results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='🌱 AgroWare Disease Prediction')

    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--crop', default=None, help='Filter predictions by crop name (e.g., Tomato, Wheat)')
    parser.add_argument('--model', default=None, help='Path to Keras model')
    parser.add_argument('--tflite', default=None, help='Path to TFLite model')
    parser.add_argument('--labels', default=DEFAULT_LABELS_PATH)

    args = parser.parse_args()

    image_path = args.image
    crop_filter = args.crop
    model_path = args.model
    tflite_path = args.tflite
    labels_path = args.labels

    # Auto-detect model
    if not model_path and not tflite_path:
        possible_models = [
            "agroware_model.h5",
            "agroware_model.keras",
            "final_model.keras",
            DEFAULT_TFLITE_PATH
        ]

        for m in possible_models:
            if os.path.exists(m):
                if m.endswith(".tflite"):
                    tflite_path = m
                    print(f"🔍 Found TFLite model: {m}")
                else:
                    model_path = m
                    print(f"🔍 Found Keras model: {m}")
                break

        if not model_path and not tflite_path:
            raise FileNotFoundError("❌ No model found in directory")

    # Run Prediction
    try:
        if tflite_path:
            print("🚀 Using TFLite")
            results = predict_tflite(image_path, tflite_path, labels_path, crop_filter)
        else:
            print("🚀 Using Keras")
            results = predict_keras(image_path, model_path, labels_path, crop_filter)

        # Output
        print("\n" + "="*40)
        print(f"🎯 Results for: {os.path.basename(image_path)}")
        if crop_filter:
            print(f"🌾 Filtered by crop: {crop_filter}")
        print("="*40)

        valid = False
        for i, r in enumerate(results):
            is_valid = r['confidence'] > CONFIDENCE_THRESHOLD
            prefix = "✅" if i == 0 and is_valid else "  "
            print(f"{prefix} {r['class']} → {r['confidence']:.2%}")

            if is_valid:
                valid = True

        if not valid:
            print("\n⚠️ Low confidence prediction")

        print("="*40 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
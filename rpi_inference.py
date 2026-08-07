"""
Runs the exported SkinSense model on a Raspberry Pi.
Supports single image or a folder of images.

USAGE:
    python3 rpi_inference.py path/to/image.jpg
    python3 rpi_inference.py path/to/folder/    (processes all images in folder)
"""
import os
import sys

# ---------- Suppress ONNX Runtime warnings (environment variables) ----------
os.environ["ORT_DISABLE_GPU"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"          # Error level only
os.environ["ORT_DEVICE_DISCOVERY_DISABLE"] = "1"

# Now import modules (these do not trigger the device‑discovery warnings)
import glob
import pickle
import warnings
import numpy as np
from PIL import Image
import onnxruntime as ort

# Suppress Python warnings
warnings.filterwarnings("ignore")
ort.set_default_logger_severity(3)

# ---------- Helpers to suppress/restore stderr (file descriptor level) ----------
_saved_stderr = None

def suppress_stderr():
    """Redirect stderr to /dev/null (silences C‑level logs)."""
    global _saved_stderr
    _saved_stderr = os.dup(2)                     # save original fd 2
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)                           # replace fd 2 with /dev/null
    os.close(devnull)

def restore_stderr():
    """Restore stderr to its original state."""
    global _saved_stderr
    if _saved_stderr is not None:
        os.dup2(_saved_stderr, 2)                 # restore original fd 2
        os.close(_saved_stderr)
        _saved_stderr = None

# ---------- Rest of the script ----------
IMAGE_SIZE = 224
MEAN = np.array([0.763, 0.546, 0.570], dtype=np.float32)
STD = np.array([0.141, 0.152, 0.169], dtype=np.float32)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def find_model_file():
    matches = glob.glob('*_int8_rpi.onnx')
    if not matches:
        raise FileNotFoundError(
            "No *_int8_rpi.onnx file found next to this script. "
            "Copy the 'rpi_export' folder from your desktop here first."
        )
    return matches[0]


def preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)   # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def predict_single(session, label_encoder, image_path):
    input_tensor = preprocess(image_path)
    outputs = session.run(None, {'input': input_tensor})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    top_idx = int(np.argmax(probs))
    predicted_class = label_encoder.inverse_transform([top_idx])[0]
    return predicted_class, probs[top_idx], probs


def print_full_result(image_path, pred_class, conf, probs, label_encoder):
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {pred_class} ({conf*100:.1f}% confidence)")
    print("All classes:")
    for idx in np.argsort(probs)[::-1]:
        class_name = label_encoder.inverse_transform([idx])[0]
        print(f"  {class_name}: {probs[idx]*100:.1f}%")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 rpi_inference.py path/to/image.jpg")
        print("   or: python3 rpi_inference.py path/to/folder/")
        sys.exit(1)

    path = sys.argv[1]

    # Load label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    model_path = find_model_file()

    # -------------------- SUPPRESS NOISY DEVICE‑DISCOVERY WARNINGS --------------------
    suppress_stderr()
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    restore_stderr()
    # --------------------------------------------------------------------------------

    if os.path.isdir(path):
        image_files = []
        for ext in IMAGE_EXTS:
            image_files.extend(glob.glob(os.path.join(path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(path, f'*{ext.upper()}')))
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No image files found in '{path}'")
            sys.exit(1)

        print(f"Processing {len(image_files)} images...")
        for img_path in image_files:
            pred_class, conf, probs = predict_single(session, label_encoder, img_path)
            print_full_result(img_path, pred_class, conf, probs, label_encoder)
    else:
        pred_class, conf, probs = predict_single(session, label_encoder, path)
        print_full_result(path, pred_class, conf, probs, label_encoder)


if __name__ == '__main__':
    main()
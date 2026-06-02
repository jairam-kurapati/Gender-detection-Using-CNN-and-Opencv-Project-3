import os
from pathlib import Path

MODEL_PATHS = [
    Path("models/gender_cnn.h5"),
    Path("Models/gender_cnn.h5")
]


def _find_model():
    for p in MODEL_PATHS:
        if p.exists():
            return str(p)
    return None


def predict_image(image_path):
    model_file = _find_model()
    if model_file is None:
        raise FileNotFoundError("No trained model found. Run training to produce 'gender_cnn.h5' in models/ or Models/.")


    try:
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError:
        raise ModuleNotFoundError("tensorflow is not installed. Install dependencies with: pip install -r requirements.txt")

    try:
        import numpy as np
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Required package missing: {exc.name}. Install with: pip install -r requirements.txt")

    model = load_model(model_file)

    # Read image with OpenCV
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read uploaded image.")

    # Detect faces using haarcascade shipped with OpenCV data
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Use the first detected face
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]

    # Preprocess to model expected size
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # Predict
    pred = model.predict(face)[0][0]

    # Interpret prediction
    # Assumes binary output with sigmoid: value near 1 -> class 1, near 0 -> class 0
    confidence = float(pred) * 100.0
    if pred >= 0.5:
        gender = "Male"
    else:
        gender = "Female"

    return gender, round(confidence, 2)


if __name__ == "__main__":
    print("This module provides `predict_image(image_path)` to run gender prediction.")

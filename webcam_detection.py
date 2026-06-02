import time
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from face_detector import detect_faces
from user_database import list_users, register_user, recognize_user

MODEL_PATHS = [
    Path("models/gender_cnn.h5"),
    Path("Models/gender_cnn.h5")
]

model_path = next((p for p in MODEL_PATHS if p.exists()), None)
if model_path is None:
    raise FileNotFoundError(
        "Trained model not found. Create 'gender_cnn.h5' in models/ by running train.py."
    )

model = load_model(str(model_path))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera connection.")

capture_dir = Path("captures")
capture_dir.mkdir(exist_ok=True)
capture_count = len(list(capture_dir.glob("capture_*.jpg")))

registered_users = list_users()
print("Webcam gender detection started.")
print(f"Registered users: {registered_users if registered_users else 'none'}")
print("Press 'r' to register a user, 'c' to capture a frame, 'q' to quit.")

registration_mode = False
registration_name = None
registration_samples = []
status_message = ""
status_expires = 0.0
last_sample_time = 0.0
sample_interval = 0.6


def draw_label(frame, text, x, y, color=(0, 255, 255)):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    overlay_y = 30

    if len(faces) == 0:
        draw_label(frame, "No face detected. Please place your face in front of the camera.", 10, overlay_y, (0, 255, 255))
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            try:
                face_input = cv2.resize(face, (128, 128))
                face_norm = face_input.astype("float32") / 255.0
                face_norm = np.expand_dims(face_norm, axis=0)

                prediction = float(model.predict(face_norm, verbose=0)[0][0])
                if prediction >= 0.5:
                    gender = "Male"
                    confidence = prediction * 100.0
                else:
                    gender = "Female"
                    confidence = (1.0 - prediction) * 100.0

                user_name, user_conf = recognize_user(face)
                if user_name == "Unknown":
                    name_text = "Name: Unknown"
                else:
                    name_text = f"Name: {user_name} ({user_conf:.0f}%)"

                gender_text = f"Gender: {gender} ({confidence:.2f}%)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_label(frame, name_text, x, y - 30, (0, 255, 0))
                draw_label(frame, gender_text, x, y - 10, (0, 255, 0))

                print(f"Detected face: {name_text}, {gender_text}")
            except Exception as exc:
                print(f"Prediction error: {exc}")

    now = time.perf_counter()
    if registration_mode:
        if len(faces) > 0 and now - last_sample_time >= sample_interval:
            x2, y2, w2, h2 = faces[0]
            face_img = frame[y2:y2+h2, x2:x2+w2]
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (128, 128))
            registration_samples.append(face_gray)
            last_sample_time = now
            status_message = f"Captured sample {len(registration_samples)} of 5."
            status_expires = now + 2.0
            print(status_message)

        if len(registration_samples) >= 5:
            saved = register_user(registration_name, registration_samples)
            status_message = f"Registration complete: saved {saved} samples for '{registration_name}'."
            status_expires = now + 4.0
            print(status_message)
            registered_users = list_users()
            registration_mode = False
            registration_name = None
            registration_samples = []
            last_sample_time = 0.0

    status_line = "Press 'r' to register, 'c' to capture, 'q' to quit."
    if registration_mode:
        status_line = f"Registering {registration_name}: {len(registration_samples)}/5. {status_message}"
    elif status_message and now <= status_expires:
        status_line = status_message

    draw_label(frame, status_line, 10, frame.shape[0] - 20, (255, 255, 255))

    cv2.imshow("Gender Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord("c") and len(faces) > 0:
        capture_count += 1
        filename = capture_dir / f"capture_{capture_count:03d}.jpg"
        cv2.imwrite(str(filename), frame)
        status_message = f"Saved capture to {filename}."
        status_expires = now + 4.0
        print(status_message)

    if key == ord("r") and not registration_mode:
        name = input("Enter new user name: ").strip()
        if not name:
            print("Registration cancelled: name cannot be empty.")
            status_message = "Registration cancelled."
            status_expires = now + 3.0
            continue
        registration_mode = True
        registration_name = name
        registration_samples = []
        status_message = "Registration started. Please hold still in front of the camera."
        status_expires = now + 4.0
        print(status_message)
        last_sample_time = 0.0

cap.release()
cv2.destroyAllWindows()
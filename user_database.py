import json
from pathlib import Path

import cv2
import numpy as np

USER_FACES_DIR = Path("user_faces")
DB_FILE = Path("user_db.json")
IMG_SIZE = (128, 128)

orb = cv2.ORB_create(nfeatures=500)
# knnMatch(k=2) is used below, so crossCheck must be disabled.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _ensure_dirs():
    USER_FACES_DIR.mkdir(exist_ok=True)


def _load_db():
    _ensure_dirs()
    if DB_FILE.exists():
        try:
            data = json.loads(DB_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    users = [entry.get("name") for entry in data.get("users", []) if entry.get("name")]
    user_dirs = [p.name for p in USER_FACES_DIR.iterdir() if p.is_dir()]

    for user_dir in user_dirs:
        if user_dir not in users:
            users.append(user_dir)

    users = sorted(set(users))
    data["users"] = [{"name": name} for name in users]
    DB_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return users


def list_users():
    return _load_db()


_templates_cache = None


def _load_face_templates(reload=False):
    global _templates_cache
    if _templates_cache is not None and not reload:
        return _templates_cache

    users = _load_db()
    templates = {}

    for user in users:
        user_dir = USER_FACES_DIR / user
        if not user_dir.exists():
            continue

        descriptors = []
        for image_path in sorted(user_dir.glob("*.jpg")) + sorted(user_dir.glob("*.png")):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            _, des = orb.detectAndCompute(img, None)
            if des is not None:
                descriptors.append(des)

        if descriptors:
            templates[user] = descriptors

    _templates_cache = templates
    return templates


def recognize_user(face_image):
    templates = _load_face_templates()
    if not templates:
        return "Unknown", 0.0

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    _, probe_des = orb.detectAndCompute(gray, None)
    if probe_des is None:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = 0.0
    best_count = 0
    second_best_score = 0.0

    for user, descriptors_list in templates.items():
        user_best = 0.0
        user_best_count = 0
        for stored_des in descriptors_list:
            if stored_des is None or probe_des is None or len(stored_des) < 2 or len(probe_des) < 2:
                continue
            matches = bf.knnMatch(probe_des, stored_des, k=2)
            if not matches:
                continue

            good_matches = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if not good_matches:
                continue

            match_ratio = len(good_matches) / max(1, len(matches))
            if len(good_matches) >= 15 and match_ratio >= 0.25:
                if match_ratio > user_best:
                    user_best = match_ratio
                    user_best_count = len(good_matches)
        if user_best > best_score:
            second_best_score = best_score
            best_score = user_best
            best_count = user_best_count
            best_name = user
        elif user_best > second_best_score:
            second_best_score = user_best

    if best_score < 0.30 or best_count < 20 or best_score - second_best_score < 0.10:
        return "Unknown", 0.0

    return best_name, min(100.0, best_score * 100.0)


def register_user(name, face_images):
    if not name:
        raise ValueError("User name cannot be empty.")

    _ensure_dirs()
    user_dir = USER_FACES_DIR / name
    user_dir.mkdir(exist_ok=True)

    existing = sorted(user_dir.glob("*.jpg")) + sorted(user_dir.glob("*.png"))
    start_index = len(existing) + 1

    saved = 0
    for i, face_image in enumerate(face_images, start=start_index):
        if face_image is None:
            continue
        if face_image.ndim == 3 and face_image.shape[2] == 3:
            img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            img = face_image
        img = cv2.resize(img, IMG_SIZE)
        path = user_dir / f"{name}_{i:03d}.jpg"
        cv2.imwrite(str(path), img)
        saved += 1

    _load_db()
    global _templates_cache
    _templates_cache = None
    return saved

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
SIZES = {
    'train': 50,
    'validation': 10,
    'test': 10
}
CLASSES = ['male', 'female']
IMG_SIZE = (128, 128)

os.makedirs(BASE_DIR, exist_ok=True)

skin_tones = [
    (244, 194, 157),
    (224, 172, 105),
    (198, 134, 66),
    (255, 219, 172)
]

hair_colors = [
    (30, 24, 16),
    (80, 50, 20),
    (120, 80, 40),
    (200, 120, 70)
]

random.seed(42)
np.random.seed(42)

def make_face(seed, gender):
    random.seed(seed)
    np.random.seed(seed)
    img = Image.new('RGB', IMG_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Background slight tint
    bg = tuple([int(240 + random.randint(-10,10))]*3)
    draw.rectangle([0,0,IMG_SIZE[0],IMG_SIZE[1]], fill=bg)

    # Face ellipse
    skin = random.choice(skin_tones)
    cx, cy = IMG_SIZE[0]//2, IMG_SIZE[1]//2 - 5
    rx, ry = 36 + random.randint(-6,6), 48 + random.randint(-6,6)
    bbox = [cx-rx, cy-ry, cx+rx, cy+ry]
    draw.ellipse(bbox, fill=skin)

    # Eyes
    eye_y = cy - 10 + random.randint(-3,3)
    eye_x_off = 16
    eye_r = 4 + random.randint(-1,1)
    draw.ellipse([cx-eye_x_off-eye_r, eye_y-eye_r, cx-eye_x_off+eye_r, eye_y+eye_r], fill=(20,20,20))
    draw.ellipse([cx+eye_x_off-eye_r, eye_y-eye_r, cx+eye_x_off+eye_r, eye_y+eye_r], fill=(20,20,20))

    # Nose
    nose_top = (cx, cy-2)
    draw.polygon([(cx, cy-2), (cx-4, cy+6), (cx+4, cy+6)], fill=(150, 100, 80))

    # Mouth
    mouth_y = cy + 14 + random.randint(-2,2)
    mouth_w = 14 + random.randint(-3,3)
    draw.arc([cx-mouth_w, mouth_y-4, cx+mouth_w, mouth_y+6], start=0, end=180, fill=(150,30,30), width=2)

    # Hair
    hair = random.choice(hair_colors)
    hair_top = cy - ry - 6
    if gender == 'female':
        # longer hair
        draw.rectangle([cx- rx - 4, hair_top, cx+rx+4, cy+10], fill=hair)
    else:
        # short hair cap
        draw.ellipse([cx-rx-4, hair_top, cx+rx+4, cy+2], fill=hair)
        # small beard for some males
        if random.random() < 0.6:
            beard_col = (max(0, hair[0]-20), max(0,hair[1]-20), max(0,hair[2]-20))
            draw.ellipse([cx-18, cy+20, cx+18, cy+32], fill=beard_col)

    # Add slight variation and blur
    arr = np.array(img).astype(np.uint8)
    noise = (np.random.randn(*arr.shape) * 6).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2,1.2)))

    return img


def ensure_dirs():
    for split in SIZES.keys():
        for cls in CLASSES:
            d = os.path.join(BASE_DIR, split, cls)
            os.makedirs(d, exist_ok=True)


def generate():
    ensure_dirs()
    for split, count in SIZES.items():
        for cls in CLASSES:
            outdir = os.path.join(BASE_DIR, split, cls)
            # If images already exist, skip generation for that class/split
            existing = len([n for n in os.listdir(outdir) if n.lower().endswith(('.png','.jpg','.jpeg'))])
            if existing >= count:
                print(f"Skipping {split}/{cls}, already has {existing} images")
                continue
            for i in range(count):
                idx = existing + i
                img = make_face(seed=hash((split,cls,idx)) & 0xffffffff, gender=cls)
                fname = f"{cls}_{idx:03d}.jpg"
                img.save(os.path.join(outdir, fname), quality=85)
            print(f"Generated {count} images for {split}/{cls}")

if __name__ == '__main__':
    generate()
    print('Done.')

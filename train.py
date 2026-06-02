import os
import sys
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        Dense,
        Flatten,
        Dropout,
        BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ModuleNotFoundError as exc:
    missing_package = exc.name
    print(f"Error: missing required package '{missing_package}'.")
    print("Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# ==========================================
# Paths
# ==========================================

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

MODEL_DIR = "models"
OUTPUT_DIR = "static/outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Image Parameters
# ==========================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# ==========================================
# Data Augmentation
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ==========================================
# CNN Model
# ==========================================

model = Sequential()

# Block 1
model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
)
model.add(MaxPooling2D(2, 2))

# Block 2
model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2, 2))

# Block 3
model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2, 2))

model.add(BatchNormalization())

# Fully Connected Layers
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/gender_cnn.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# ==========================================
# Save Final Model
# ==========================================

model.save("models/gender_cnn.h5")

print("Model Saved Successfully!")

# ==========================================
# Accuracy Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train Accuracy",
    "Validation Accuracy"
])

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy.png"
    )
)

# ==========================================
# Loss Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Train Loss",
    "Validation Loss"
])

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss.png"
    )
)

print("Graphs Saved Successfully!")
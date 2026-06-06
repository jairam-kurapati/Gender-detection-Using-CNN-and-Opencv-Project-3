# Gender Detection System Using CNN and OpenCV

## Overview

The Gender Detection System is a Deep Learning-based Computer Vision application that predicts whether a person is Male or Female from a facial image. The system uses a Convolutional Neural Network (CNN) for feature extraction and classification and OpenCV for image processing and face detection.

The project provides a user-friendly web interface where users can upload an image and receive gender prediction results in real time.

---

# Features

* Face Detection using OpenCV
* Gender Classification using CNN
* Image Upload through Web Interface
* Real-Time Prediction
* Confidence Score Display
* User-Friendly Dashboard
* Deep Learning-Based Classification
* Responsive Frontend Design

---

# Technologies Used

## Programming Language

* Python

## Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Network (CNN)

## Computer Vision

* OpenCV

## Data Processing

* NumPy
* Matplotlib

## Web Development

* Flask
* HTML
* CSS
* JavaScript

---

# Project Architecture

Image Upload
↓
OpenCV Face Detection
↓
Face Extraction
↓
Image Preprocessing
↓
CNN Model
↓
Feature Extraction
↓
Gender Classification
↓
Prediction Result

---

# Dataset

The model is trained on a facial image dataset containing:

* Male Images
* Female Images

Dataset Structure:

dataset/

├── train/

│ ├── male/

│ └── female/

│

└── test/

├── male/

└── female/

---

# CNN Architecture

The Convolutional Neural Network consists of:

### Input Layer

* Image Size: 128 × 128 × 3

### Convolution Layer 1

* 32 Filters
* ReLU Activation

### Max Pooling Layer

* Pool Size 2×2

### Convolution Layer 2

* 64 Filters
* ReLU Activation

### Max Pooling Layer

* Pool Size 2×2

### Convolution Layer 3

* 128 Filters
* ReLU Activation

### Max Pooling Layer

* Pool Size 2×2

### Flatten Layer

Converts feature maps into a vector.

### Dense Layer

* 128 Neurons
* ReLU Activation

### Output Layer

* Sigmoid Activation

Output:

* Male
* Female

---

# Working Process

## Step 1: Image Upload

User uploads a facial image through the web application.

## Step 2: Face Detection

OpenCV detects the face region.

```python
face = detector.detectMultiScale(
    gray,
    1.3,
    5
)
```

## Step 3: Preprocessing

The face image is:

* Cropped
* Resized
* Normalized

Image Size:

128 × 128

## Step 4: CNN Prediction

The CNN extracts facial features and predicts gender.

## Step 5: Result Display

Output Example:

Gender: Male

Confidence: 96.45%

---

# Project Structure

Gender_Detection_System/

├── app.py

├── train_model.py

├── requirements.txt

├── model/

│ └── gender_model.h5

│

├── dataset/

│ ├── train/

│ └── test/

│

├── static/

│ ├── css/

│ ├── js/

│ └── uploads/

│

├── templates/

│ ├── index.html

│ └── result.html

│

└── README.md

---

# Installation

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install tensorflow
pip install opencv-python
pip install flask
pip install numpy
pip install matplotlib
```

Or

```bash
pip install -r requirements.txt
```

---

# Training the Model

Run:

```bash
python train_model.py
```

The trained model will be saved as:

```text
model/gender_model.h5
```

---

# Running the Application

Start Flask server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Output

Input:

* Facial Image

Output:

* Predicted Gender
* Confidence Score

Example:

Gender: Female

Confidence: 98.12%

---

# Algorithms Used

## Haar Cascade Classifier

Purpose:

* Face Detection

Library:

* OpenCV

## Convolutional Neural Network (CNN)

Purpose:

* Feature Extraction
* Gender Classification

## ReLU Activation

Purpose:

* Non-linear learning

## Max Pooling

Purpose:

* Dimensionality Reduction

## Sigmoid Activation

Purpose:

* Binary Classification

---

# Applications

* Smart Surveillance Systems
* Human Computer Interaction
* Demographic Analytics
* Retail Customer Analysis
* Attendance Systems
* AI-Based Security Systems

---

# Future Enhancements

* Age Detection
* Emotion Recognition
* Multi-Face Gender Detection
* Real-Time Webcam Prediction
* Mobile Application Deployment
* Cloud Deployment

---

# Resume Description

Developed a Deep Learning-based Gender Detection System using Convolutional Neural Networks (CNN), TensorFlow, and OpenCV. Implemented face detection, image preprocessing, feature extraction, and gender classification with a responsive Flask web interface for real-time predictions.

---

# Author

Developed as an AIML Project using Deep Learning and Computer Vision technologies.

# Simulation & Deep Learning Pipeline (Webots & TensorFlow) 

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Webots](https://img.shields.io/badge/Webots-2023-brightgreen?logo=webots&logoColor=white)](https://cyberbotics.com/)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/yourusername/Webots-TensorFlow-Pipeline)

---

## **Project Overview**

This project demonstrates a **complete simulation-to-AI pipeline** where a **3D robot simulation in Webots** generates **synthetic image datasets** for **deep learning training**, and a **Convolutional Neural Network (CNN)** classifies objects in **real-time using OpenCV**.

The workflow integrates:  

1. **Robot Simulation:** Create a virtual 3D environment with a robot equipped with a camera.  
2. **Dataset Generation:** Capture synthetic images of objects for training.  
3. **Deep Learning:** Train a CNN using TensorFlow to recognize objects.  
4. **Real-Time Inference:** Use OpenCV to classify objects from a live camera feed.

This project demonstrates skills in **robotics, simulation, computer vision, deep learning, and Python programming**, making it ideal for internships in **AI, robotics, and computer vision domains**.

---

## **Key Features**

- **3D Robot Simulation (Webots):**  
  - Robot navigates a virtual environment.  
  - Camera captures synthetic images for dataset generation.  

- **Dataset Generation:**  
  - Automated image collection of objects from multiple angles.  
  - Labels stored for supervised learning.  

- **CNN Model (TensorFlow):**  
  - Convolutional layers for feature extraction.  
  - Fully connected layers for classification.  
  - Supports multi-class object detection.  

- **Real-Time Prediction (OpenCV):**  
  - Processes live camera input.  
  - Displays predicted class on video feed.  

- **Modular Design:**  
  - Easily add new objects, environments, or models.  
  - Scalable pipeline for larger datasets or more complex tasks.  

---

## **Tech Stack**

| Category              | Technology / Library                  |
|-----------------------|--------------------------------------|
| Simulation            | Webots                               |
| Programming Language  | Python 3                             |
| Deep Learning         | TensorFlow 2.x                        |
| Computer Vision       | OpenCV                               |
| Data Handling         | NumPy, Pandas                         |
| Visualization         | Matplotlib                            |


## **Detailed Workflow**

### **1. Simulation (Webots)**

- Robot equipped with a virtual camera navigates the environment.  
- The environment contains multiple objects for classification.  
- Robot moves automatically or via a controller script.  
- Images are captured frame-by-frame and stored in a dataset folder.

### **2. Dataset Generation**

- Captured images are labeled based on object type.  
- Images are preprocessed (resized, normalized) for CNN training.  
- Dataset split: **80% training, 20% validation**.  

### **3. CNN Training (TensorFlow)**

- CNN architecture includes:  
  - 2–3 convolutional layers with ReLU activation  
  - MaxPooling layers for dimensionality reduction  
  - Fully connected dense layers  
  - Softmax output layer for multi-class classification  
- Training includes **data augmentation** for robustness.  
- Model is saved as `cnn_model.h5` for inference.  

```python
# Sample training snippet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


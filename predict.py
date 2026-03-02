# predict.py
import cv2
import tensorflow as tf
import numpy as np
import pickle
import os

# Load trained CNN model
model_path = "../cnn_model/cnn_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Load label encoder
le_path = "../cnn_model/label_encoder.pkl"
if not os.path.exists(le_path):
    raise FileNotFoundError(f"Label encoder not found at {le_path}")

with open(le_path, "rb") as f:
    le = pickle.load(f)
print("Label encoder loaded successfully.")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Starting real-time object prediction. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess frame for CNN
    img = cv2.resize(frame, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict object class
    pred_probs = model.predict(img)
    pred_class = np.argmax(pred_probs)
    class_name = le.inverse_transform([pred_class])[0]

    # Display prediction on frame
    cv2.putText(
        frame,
        f"Prediction: {class_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time Object Classification", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

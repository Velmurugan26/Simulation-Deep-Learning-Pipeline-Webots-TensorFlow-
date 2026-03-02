# train_cnn.py
import tensorflow as tf
from model import create_cnn
from utils import load_dataset
from sklearn.preprocessing import LabelEncoder

# Load dataset
X, y = load_dataset()

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create CNN model
num_classes = len(np.unique(y_encoded))
model = create_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# Save trained model
model.save("cnn_model.h5")
print("Model training complete. Saved as cnn_model.h5")

# Optional: save label encoder for inference
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved as label_encoder.pkl")

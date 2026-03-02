# utils.py
import os
import cv2
import numpy as np
import pandas as pd

def load_dataset(dataset_path="../dataset/", img_size=(64,64)):
    """
    Load images and labels from the dataset folder.

    Args:
        dataset_path (str): Path to dataset folder
        img_size (tuple): Resize images to this size

    Returns:
        X (np.array): Image data
        y (np.array): Corresponding labels
    """
    labels_file = os.path.join(dataset_path, "labels.csv")
    df = pd.read_csv(labels_file)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(dataset_path, "images", row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(row['label'])

    # Convert to NumPy arrays
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)

    return X, y

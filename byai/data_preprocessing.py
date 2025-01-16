import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_labels(data_path, label_path):
    # Load images and labels from given paths
    images = []
    labels = []
    
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(data_path, filename)
            image = cv2.imread(image_path)
            images.append(image)

            # Assuming corresponding labels are stored in a separate label file
            label_file = os.path.join(label_path, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            with open(label_file, 'r') as f:
                label = int(f.readline().strip())  # Assuming label is a single integer (e.g., 0 for benign, 1 for malignant)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def augment_data(images):
    # Image Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(images)

def prepare_data(data_path, label_path, test_size=0.2):
    images, labels = load_images_and_labels(data_path, label_path)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    
    # Augment training data
    augmented_data = augment_data(X_train)
    
    return X_train, X_test, y_train, y_test, augmented_data

import numpy as np
import tensorflow as tf

def preprocess_images(images):
    # Normalize images
    images = images / 255.0
    # Resize images to 224x224 and convert to RGB
    images = tf.image.resize(images, [224, 224])
    images = tf.image.grayscale_to_rgb(tf.expand_dims(images, axis=-1))
    return images

def preprocess_dataset(data):
    # Extract images and labels
    x_train = data['train_images']
    y_train = data['train_labels']
    x_val = data['val_images']
    y_val = data['val_labels']
    x_test = data['test_images']
    y_test = data['test_labels']

    # Preprocess images
    x_train = preprocess_images(x_train)
    x_val = preprocess_images(x_val)
    x_test = preprocess_images(x_test)

    return {
        'train_images': x_train,
        'train_labels': y_train,
        'val_images': x_val,
        'val_labels': y_val,
        'test_images': x_test,
        'test_labels': y_test
    }

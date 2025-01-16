# import numpy as np

# # Load the .npz file
# data = np.load("data/breastmnist.npz")

# # Print the keys in the .npz file to see what is available
# print(data.files)
import tensorflow as tf

# Set up TensorFlow to use the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')  # Choose the first GPU
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Dynamically allocate memory
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")

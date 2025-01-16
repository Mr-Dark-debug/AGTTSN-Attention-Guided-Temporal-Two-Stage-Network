import requests
import os
import numpy as np

def download_and_extract_dataset():
    url = 'https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/main/data/breastmnist.npz'
    file_path = 'breastmnist.npz'

    # Download the dataset if it doesn't exist locally
    if not os.path.exists(file_path):
        print("Downloading BreastMNIST dataset...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists locally.")

    # Load the dataset
    data = np.load(file_path)
    return data

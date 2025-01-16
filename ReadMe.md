# Active Learning with Deep Active Networks for Breast Cancer Detection

## Overview

This repository contains a proof-of-concept implementation of a deep learning-based active learning pipeline for breast cancer detection using the BreastMNIST dataset. The pipeline utilizes two key networks: a **D-Net (Detection Network)** for bounding box prediction and a **C-Net (Classification Network)** for binary classification (benign or malignant). 

The project demonstrates the use of **active learning** to iteratively select uncertain samples from the dataset for model training. The key goal is to explore how active learning can be used to reduce the annotation burden while improving the model's accuracy in detecting breast cancer.

## Project Architecture

The architecture involves the following components:

### 1. **D-Net (Detection Network)**:
   - **Purpose**: The D-Net is designed to predict the bounding boxes for breast tissue within the input images. It uses a **ResNet50** backbone and is trained to predict four values representing the bounding box coordinates.
   - **Model Type**: Convolutional Neural Network (CNN) built on top of ResNet50 with a custom head to predict bounding boxes.
   - **Output**: 4 values (x_min, y_min, x_max, y_max) representing the coordinates of the bounding box.

### 2. **C-Net (Classification Network)**:
   - **Purpose**: The C-Net is designed to perform binary classification, predicting whether the image corresponds to a benign or malignant tumor. It uses the ResNet50 backbone and is trained with a binary cross-entropy loss function.
   - **Model Type**: CNN with ResNet50 backbone, ending with a dense layer and sigmoid activation for binary classification.
   - **Output**: A single value representing the probability of being malignant (1 = malignant, 0 = benign).

### 3. **CoreSet Sampling (Active Learning)**:
   - **Purpose**: CoreSet Sampling is used to select the most uncertain samples in the dataset for retraining. This helps reduce the number of samples required for labeling, focusing only on those that are uncertain or likely to improve the model.
   - **Mechanism**: The CoreSet Sampling algorithm selects samples based on their distance from other samples in the embedding space. The more distant a sample is from others, the more uncertain the model is about that sample.

### 4. **Self-Distillation**:
   - **Purpose**: Self-distillation is used to enhance the performance of the classification network (C-Net). The C-Net is trained in two phases:
     - **Hard Labels**: Traditional classification loss using the true labels.
     - **Soft Labels**: The predictions made by the teacher model (another instance of the same C-Net) are used as "soft" targets, helping the student network learn better.
   - **Temperature Scaling**: Softmax logits are scaled with a temperature parameter to smooth the predictions and allow better generalization.

## Dataset

The dataset used is the **BreastMNIST** dataset, which is a variant of the MNIST dataset with breast cancer-related images.

- **Images**: Grayscale 28x28 images of breast tissue samples.
- **Labels**: Binary labels (1 for malignant, 0 for benign).
- **Bounding Boxes**: Coordinates of regions of interest (ROIs) in the images containing breast tissue. If bounding boxes are unavailable, the code generates dummy bounding boxes.

The dataset is expected to be stored in `.npz` format with the following keys:
- `train_images`: The images for training.
- `train_labels`: The labels corresponding to the images.
- `train_bboxes`: The bounding box coordinates for the training images (optional, will generate dummy boxes if not available).

## Workflow

The active learning pipeline follows these steps:

1. **Load Data**: The dataset is loaded, and images are resized to 224x224 pixels.
2. **Model Training**:
   - Initially, the D-Net and C-Net are trained on the full dataset.
   - The C-Net is used to make predictions on the dataset, and the D-Net predicts the bounding boxes for the images.
3. **CoreSet Sampling**: A small set of uncertain samples (based on the embeddings from C-Net) is selected for further training.
4. **Self-Distillation**: The selected samples are used to train the C-Net with self-distillation, where the teacher model provides soft labels to the student model.
5. **Model Evaluation**: The model's accuracy is evaluated after each iteration to monitor performance.

This iterative process continues for several active learning iterations, with each cycle focusing on improving the model's performance by training it on uncertain samples.

## Requirements

The following libraries and frameworks are required for running the code:

- TensorFlow (v2.x)
- Numpy
- SciPy
- scikit-learn

You can install the dependencies using `pip`:

```bash
pip install tensorflow numpy scipy scikit-learn
```

## Code Structure

- **`load_breastmnist_data()`**: Loads the BreastMNIST dataset from the `.npz` file and preprocesses the images.
- **`create_dnet()`**: Defines and compiles the D-Net model.
- **`create_cnet()`**: Defines and compiles the C-Net model and its embedding model.
- **`coreset_sampling()`**: Selects the most uncertain samples from the dataset based on CoreSet sampling.
- **`self_distillation()`**: Implements the self-distillation process to improve the C-Net using the teacher model.
- **`active_learning_pipeline()`**: Manages the active learning pipeline, including model training and evaluation.

## Training the Model

You can train the model by running the script:

```bash
python main.py
```

This will start the active learning pipeline, train the D-Net and C-Net, and perform the CoreSet sampling and self-distillation in an iterative manner.

## Evaluation

After each iteration, the model's accuracy is evaluated using the classification network (C-Net). The accuracy is reported for each active learning cycle, allowing you to observe improvements as the model is trained on uncertain samples.

The evaluation includes:
- **Training the models**: The D-Net and C-Net are trained iteratively using CoreSet sampling, where only the most uncertain samples are selected.
- **Self-Distillation**: The C-Net is improved via self-distillation, where predictions made by a teacher model guide the training of the student model.
- **Accuracy Calculation**: After every iteration, the accuracy of the C-Net is calculated based on the selected training samples.

## Future Work

- **Improving CoreSet Sampling**: Experiment with different active learning sampling strategies to select even more uncertain samples. Techniques like uncertainty sampling, query-by-committee, or Bayesian active learning could be explored to enhance the learning process.
- **Hyperparameter Tuning**: Explore different hyperparameters for the models such as learning rate, batch size, and temperature scaling in distillation. Adjusting these hyperparameters could improve the performance of both D-Net and C-Net.
- **Bounding Box Prediction**: Enhance the D-Net to be more accurate in predicting the bounding boxes, potentially improving the detection and localization of tumors for better predictions and model performance.
- **Model Interpretability**: Investigate methods for explaining the decisions made by the models, especially for medical applications where model interpretability is crucial for understanding predictions.

## Contributing

Contributions are welcome! If you find any issues or want to add features, feel free to create a pull request or open an issue. Here are some areas where contributions are encouraged:
- Enhancing the active learning algorithm to improve sample selection.
- Experimenting with different architectures and loss functions for better performance.
- Adding more robust error handling and model evaluation metrics.
- Writing unit tests and improving documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

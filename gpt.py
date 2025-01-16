import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import os

# Load the breastmnist dataset (without bounding boxes)
def load_breastmnist_data(data_path="data/breastmnist.npz"):
    data = np.load(data_path)
    images = data['train_images']  # Shape: (num_samples, 28, 28)
    labels = data['train_labels']  # Shape: (num_samples,)
    
    # Check if bounding boxes exist, else generate dummy ones
    if 'train_bboxes' in data:
        bboxes = data['train_bboxes']  
    else:
        print("Bounding boxes not found! Generating dummy ones.")
        bboxes = np.zeros((images.shape[0], 4))  # Creating dummy zero bounding boxes
    
    # Normalize images and convert grayscale to RGB
    images = images / 255.0
    images = np.stack([images] * 3, axis=-1)  # Shape: (num_samples, 28, 28, 3)
    
    # Resize images to (224, 224)
    images_resized = np.array([tf.image.resize(img, (224, 224)).numpy() for img in images])
    
    # Convert labels to binary
    labels = np.expand_dims(labels, axis=-1)
    
    print(f"Images shape: {images_resized.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Bounding boxes shape: {bboxes.shape}")
    
    return images_resized, labels, bboxes

# Define the D-Net (Detection Network)
def create_dnet():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    bbox_output = layers.Dense(4, activation='sigmoid')(x)  # Output 4 values for bounding box
    return models.Model(inputs=base_model.input, outputs=bbox_output)

# Define the C-Net (Classification Network)
def create_cnet():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    class_output = layers.Dense(1, activation='sigmoid')(x)  # Output single binary classification value
    model = models.Model(inputs=base_model.input, outputs=class_output)
    
    # Embedding model for CoreSet sampling
    embedding_model = models.Model(inputs=base_model.input, outputs=x)
    return model, embedding_model

# CoreSet Sampling for Active Learning
def coreset_sampling(embeddings, n_samples):
    selected_indices = []
    min_distances = np.full(embeddings.shape[0], np.inf)
    first_idx = np.random.choice(range(embeddings.shape[0]))
    selected_indices.append(first_idx)

    for _ in range(n_samples - 1):
        dists = distance.cdist(embeddings, embeddings[selected_indices])
        min_distances = np.minimum(min_distances, np.min(dists, axis=1))
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)

    return selected_indices

# Self-Distillation Function
def self_distillation(teacher_model, student_model, data, labels, temperature=5.0, alpha=0.5):
    teacher_preds = teacher_model.predict(data)
    soft_targets = tf.nn.softmax(teacher_preds / temperature)

    # Remove singleton dimensions from labels
    labels = np.squeeze(labels)

    def distillation_loss(y_true, y_pred):
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        soft_loss = tf.keras.losses.binary_crossentropy(soft_targets, y_pred)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    student_model.compile(optimizer='adam', loss=distillation_loss, metrics=['accuracy'])
    print(f"Training student model with {data.shape[0]} samples.")
    student_model.fit(data, labels, epochs=3, batch_size=32)

# Active Learning Pipeline
def active_learning_pipeline(dnet, cnet, embedding_model, dataset, iterations=5, n_samples=100, batch_size=32):
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}")
        
        # Generate predictions
        rois = dnet.predict(dataset['images'])
        predictions = cnet.predict(dataset['images'])
        
        # Extract embeddings for CoreSet sampling
        embeddings = embedding_model.predict(dataset['images'])

        # CoreSet sampling to select uncertain samples
        uncertain_indices = coreset_sampling(embeddings, n_samples)

        # Select uncertain samples and corresponding labels and bounding boxes
        selected_images = dataset['images'][uncertain_indices]
        selected_labels = dataset['labels'][uncertain_indices]
        selected_bboxes = dataset['bboxes'][uncertain_indices]

        print(f"Selected {len(selected_images)} uncertain samples.")

        # Train D-Net on selected samples using ground truth bounding boxes
        print(f"Training D-Net on selected samples with {selected_images.shape[0]} samples.")
        dnet.fit(selected_images, selected_bboxes, epochs=3, batch_size=batch_size)

        # Self-distillation for C-Net
        print(f"Self-distillation for C-Net.")
        self_distillation(cnet, cnet, selected_images, selected_labels)

        # Evaluate accuracy
        y_pred = np.round(cnet.predict(dataset['images']))
        acc = accuracy_score(dataset['labels'], y_pred)
        print(f"Accuracy after iteration {iteration + 1}: {acc:.4f}")

def main():
    # Load dataset with bounding boxes
    images, labels, bboxes = load_breastmnist_data(data_path="data/breastmnist.npz")
    
    # Split dataset into training and testing sets
    train_size = int(0.8 * len(images))
    X_train, X_test = images[:train_size], images[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    bboxes_train, bboxes_test = bboxes[:train_size], bboxes[train_size:]
    
    print(f"Train images shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Test images shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    # Create models
    dnet = create_dnet()
    cnet, embedding_model = create_cnet()
    
    # Compile models
    dnet.compile(optimizer='adam', loss='mse')
    cnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare the dataset
    dataset = {
        'images': X_train,
        'labels': y_train,
        'bboxes': bboxes_train
    }
    
    # Run the active learning pipeline
    active_learning_pipeline(dnet, cnet, embedding_model, dataset)

if __name__ == "__main__":
    main()

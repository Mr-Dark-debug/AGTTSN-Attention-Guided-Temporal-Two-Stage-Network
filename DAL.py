import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

# Define the D-Net (Detection Network) based on ResNet50
def create_dnet():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    bbox_output = layers.Dense(4, activation='sigmoid')(x)  # Bounding box regression
    return models.Model(inputs=base_model.input, outputs=bbox_output)

# Define the C-Net (Classification Network) based on ResNet50
def create_cnet():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    class_output = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
    return models.Model(inputs=base_model.input, outputs=class_output)

# Prepare the D-Net and C-Net models
dnet = create_dnet()
cnet = create_cnet()
dnet.compile(optimizer='adam', loss='mse')
cnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

    def distillation_loss(y_true, y_pred):
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        soft_loss = tf.keras.losses.binary_crossentropy(soft_targets, y_pred)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    student_model.compile(optimizer='adam', loss=distillation_loss, metrics=['accuracy'])
    student_model.fit(data, labels, epochs=3, batch_size=32)

# Active Learning Pipeline combining Self-Distillation and CoreSet
def active_learning_pipeline(dnet, cnet, dataset, iterations=5, n_samples=100):
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        rois = dnet.predict(dataset['images'])
        predictions = cnet.predict(dataset['images'])
        embeddings = cnet.layers[-2].output
        uncertain_indices = coreset_sampling(embeddings, n_samples)

        selected_images = dataset['images'][uncertain_indices]
        selected_labels = dataset['labels'][uncertain_indices]

        dnet.fit(selected_images, dataset['bboxes'][uncertain_indices], epochs=3)
        self_distillation(cnet, cnet, selected_images, selected_labels)

        y_pred = np.round(cnet.predict(dataset['images']))
        acc = accuracy_score(dataset['labels'], y_pred)
        print(f"Accuracy after iteration {iteration + 1}: {acc:.4f}")

# Simulating a medical dataset for the pipeline
dataset = {
    'images': np.random.rand(1000, 224, 224, 3),
    'labels': np.random.randint(0, 2, 1000),
    'bboxes': np.random.rand(1000, 4)
}

# Run the active learning pipeline
active_learning_pipeline(dnet, cnet, dataset)

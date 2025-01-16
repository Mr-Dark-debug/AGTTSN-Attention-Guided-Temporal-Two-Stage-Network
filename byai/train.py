import tensorflow as tf
from model import create_TSDDNet
from data_preprocessing import prepare_data
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(X_train, y_train, X_test, y_test, augmented_data, epochs=50):
    model = create_TSDDNet()
    
    # First stage training: Training with partially annotated data
    print("Training Stage 1: With partially annotated data...")
    model.fit(augmented_data, epochs=epochs, validation_data=(X_test, y_test))
    
    # Second stage training: Fine-tuning with fully annotated data (if applicable)
    print("Training Stage 2: Fine-tuning with fully annotated data...")
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate performance on the test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Binarize predictions

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

def main():
    # Paths to dataset and labels
    data_path = "path_to_images"
    label_path = "path_to_labels"
    
    # Prepare data
    X_train, X_test, y_train, y_test, augmented_data = prepare_data(data_path, label_path)
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, augmented_data)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

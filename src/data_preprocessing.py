
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def load_data():
    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    print("Dataset loaded successfully!")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    # Convert images into raw pixel feature vectors
    X_train_raw = X_train.reshape(X_train.shape[0], -1)
    X_test_raw = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixel values between 0 and 1
    X_train_raw = X_train_raw.astype("float32") / 255.0
    X_test_raw = X_test_raw.astype("float32") / 255.0

    print("Preprocessing completed!")
    print("Training features:", X_train_raw.shape)
    print("Testing features:", X_test_raw.shape)

    return X_train_raw, X_test_raw


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    X_train_raw, X_test_raw = preprocess_data(
        X_train, X_test
    )

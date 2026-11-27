import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import fashion_mnist
from skimage.feature import hog

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalize raw pixel values
X_test_raw = X_test.reshape(len(X_test), 784) / 255.0

# Load models
raw_model = joblib.load("models/raw_svm.pkl")
hog_model = joblib.load("models/hog_svm.pkl")

# Extract HOG features
X_test_hog = []

for image in X_test:
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )
    X_test_hog.append(features)

X_test_hog = np.array(X_test_hog)

# Predictions
raw_predictions = raw_model.predict(X_test_raw)
hog_predictions = hog_model.predict(X_test_hog)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))

# Evaluate both models
evaluate_model("Raw-Pixel SVM", y_test, raw_predictions)
evaluate_model("HOG-SVM", y_test, hog_predictions)
import pandas as pd

results = []

for name, predictions in [
    ("Raw-Pixel SVM", raw_predictions),
    ("HOG-SVM", hog_predictions)
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted"),
        "Recall": recall_score(y_test, predictions, average="weighted"),
        "F1_Score": f1_score(y_test, predictions, average="weighted"),
        "Test_Samples": len(y_test)
    })

results_df = pd.DataFrame(results)

results_df.to_csv("results/model_comparison.csv", index=False)

print("\nComparison results saved to results/model_comparison.csv")
print(results_df)
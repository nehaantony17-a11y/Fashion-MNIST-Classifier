import joblib

# Load HOG-SVM model
hog_model = joblib.load("models/hog_svm.pkl")

# Load Raw-Pixel SVM model
raw_model = joblib.load("models/raw_svm.pkl")

print("HOG-SVM model loaded successfully!")
print("Raw-Pixel SVM model loaded successfully!")



















from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load Fashion-MNIST test dataset
(_, _), (X_test, y_test) = fashion_mnist.load_data()

# Select one test image
image = X_test[0]

# Display the image
plt.imshow(image, cmap="gray")
plt.title("Actual Label: " + str(y_test[0]))
plt.axis("off")
plt.show()
# Fashion-MNIST class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Prepare image for Raw-Pixel SVM
raw_image = image.reshape(1, 784) / 255.0

# Raw-Pixel prediction
raw_prediction = raw_model.predict(raw_image)

# Display Raw-Pixel result
print("\nRaw-Pixel SVM Prediction:", class_names[raw_prediction[0]])

# Display actual class
print("Actual Class:", class_names[y_test[0]])
# Prepare image for HOG-SVM
from skimage.feature import hog

hog_features = hog(
    image,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=False
)

# HOG prediction
hog_prediction = hog_model.predict([hog_features])

# Display HOG result
print("\nHOG-SVM Prediction:", class_names[hog_prediction[0]])
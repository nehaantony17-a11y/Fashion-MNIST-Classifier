
import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import fashion_mnist

# Fashion-MNIST class names
class_names = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot"
]

# Create output folder
output_folder = "sample_images"
os.makedirs(output_folder, exist_ok=True)

print("Loading Fashion-MNIST dataset...")

# Load original Fashion-MNIST data
(_, _), (X_test, y_test) = fashion_mnist.load_data()

print("Test dataset loaded successfully!")

# Extract one original test image from each class
for class_id, class_name in enumerate(class_names):

    # Find images belonging to the current class
    indices = np.where(y_test == class_id)[0]

    # Select the first test image of that class
    image_index = indices[0]

    # Get the original 28x28 grayscale image
    image = X_test[image_index]

    # Convert to image
    image = Image.fromarray(image.astype("uint8"))

    # Save the image
    filename = f"{class_id}_{class_name}.png"
    filepath = os.path.join(output_folder, filename)

    image.save(filepath)

    print(f"Saved: {filepath}")

print("\nAll 10 sample images created successfully!")
print(f"Images are available in the '{output_folder}' folder.")

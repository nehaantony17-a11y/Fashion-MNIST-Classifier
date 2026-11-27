
import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from datetime import datetime

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Fashion-MNIST Image Classification")
st.write(
    "Upload a clothing image to compare predictions from "
    "Raw-Pixel SVM and HOG-SVM models."
)

# ==========================================================
# 2. CLASS LABELS
# ==========================================================

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

# ==========================================================
# 3. LOAD TRAINED MODELS
# ==========================================================

@st.cache_resource
def load_models():

    hog_model = joblib.load("models/hog_svm.pkl")
    raw_model = joblib.load("models/raw_svm.pkl")

    return hog_model, raw_model


try:
    hog_model, raw_model = load_models()
    st.success("Both trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# ==========================================================
# 4. IMAGE PREPROCESSING AND PREDICTION
# ==========================================================

def predict_image(image):

    # Convert uploaded image to grayscale
    image = image.convert("L")

    # Convert to NumPy array
    image_array = np.array(image)

    # Preserve original Fashion-MNIST images (28x28)
    if image_array.shape == (28, 28):
        processed_array = image_array

    else:
        # Resize other uploaded images to 28x28
        image = image.resize((28, 28))
        processed_array = np.array(image)

    # RAW-PIXEL PREDICTION
    raw_features = processed_array / 255.0
    raw_features = raw_features.reshape(1, 784)

    raw_prediction = raw_model.predict(raw_features)[0]

    # HOG FEATURE EXTRACTION
    hog_features = hog(
        processed_array,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )

    hog_features = hog_features.reshape(1, -1)

    # HOG-SVM PREDICTION
    hog_prediction = hog_model.predict(hog_features)[0]

    return raw_prediction, hog_prediction, processed_array
  

# ==========================================================
# 5. IMAGE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG, or JPEG image."
)


# ==========================================================
# 6. DISPLAY IMAGE AND PREDICTIONS
# ==========================================================

if uploaded_file is not None:

    try:

        # Read uploaded image
        image = Image.open(uploaded_file)

        # Display original uploaded image
        st.subheader("Uploaded Image")

        st.image(
            image,
            caption=uploaded_file.name,
            width=300
        )

        # Perform predictions
        raw_prediction, hog_prediction, processed_image = predict_image(image)

        # Display preprocessed image
        st.subheader("Preprocessed Image (28 × 28)")

        st.image(
            processed_image,
            caption="Grayscale, cropped, resized and centered image",
            width=150
        )

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Raw-Pixel SVM")

            st.success(
                class_names[int(raw_prediction)]
            )

        with col2:

            st.markdown("### HOG-SVM")

            st.success(
                class_names[int(hog_prediction)]
            )

        # --------------------------------------------------
        # SAVE PREDICTIONS
        # --------------------------------------------------

        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": uploaded_file.name,
            "Raw_Pixel_Prediction": class_names[int(raw_prediction)],
            "HOG_Prediction": class_names[int(hog_prediction)]
        }

        # Results directory
        results_folder = "results"

        os.makedirs(results_folder, exist_ok=True)

        # CSV file path
        results_file = os.path.join(
            results_folder,
            "predictions.csv"
        )

        # Convert result to DataFrame
        results_df = pd.DataFrame([result])

        # Append results
        if os.path.exists(results_file):

            results_df.to_csv(
                results_file,
                mode="a",
                header=False,
                index=False
            )

        else:

            results_df.to_csv(
                results_file,
                index=False
            )

        st.success("Prediction results saved successfully!")

    except (UnidentifiedImageError, OSError):

        st.error(
            "Invalid image file. Please upload a valid image."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# ==========================================================
# 7. INITIAL MESSAGE
# ==========================================================

else:

    st.info(
        "Upload an image to begin clothing classification."
    )

    st.markdown(
        """
        **Supported image formats:**
        - PNG
        - JPG
        - JPEG

        **Models used:**
        - Raw-Pixel SVM
        - HOG-SVM
        """
    )

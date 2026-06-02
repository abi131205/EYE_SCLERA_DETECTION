import streamlit as st
import tensorflow as tf  # pylint: disable=import-error
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Early Jaundice Risk Prediction",
    page_icon="🩺",
    layout="centered"
) 

# =========================
# TITLE
# =========================
st.title("🩺 Early Jaundice Risk Prediction")
st.write("Upload an eye sclera image for jaundice prediction.")

# =========================
# MODEL PATH
# =========================
MODEL_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\models\mobilenet_jaundice_model.h5"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# IMAGE PARAMETERS
# =========================
IMG_SIZE = (224, 224)

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    # Show uploaded image
    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Eye Image",
        use_container_width=True
    )

    # Preprocess image
    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Results
    st.subheader("Prediction Result")

    if prediction > 0.5:
        confidence = prediction * 100

        st.success("✅ NORMAL")

        st.write(f"Confidence: {confidence:.2f}%")

    else:
        confidence = (1 - prediction) * 100

        st.error("⚠️ JAUNDICE DETECTED")

        st.write(f"Confidence: {confidence:.2f}%")
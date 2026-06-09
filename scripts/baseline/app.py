import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI-Powered Early Jaundice Risk Prediction",
    page_icon="👁️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("👁️ AI-Powered Early Jaundice Risk Prediction")

st.markdown("""
### SimCLR Self-Supervised Learning + MobileNetV2 + Explainable AI

Upload an eye sclera image and predict jaundice risk using our SSL-enhanced deep learning framework.
""")

# =========================
# MODEL INFORMATION
# =========================

st.info("""
### Model Information

✅ Architecture: MobileNetV2

✅ Learning Strategy: SimCLR Self-Supervised Learning

✅ Explainability: Grad-CAM
""")

# =========================
# MODEL PATH
# =========================

MODEL_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\models\experiments\ssl_finetuned_model.h5"

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

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
    "📂 Upload Eye Sclera Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image_pil = Image.open(uploaded_file)

    st.image(
        image_pil,
        caption="Uploaded Eye Sclera Image",
        use_container_width=True
    )

    # -------------------------
    # Preprocessing
    # -------------------------

    img = image_pil.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:

        confidence = prediction * 100

        st.success("✅ NORMAL")

        st.markdown("### 🎯 Confidence Score")
        st.success(f"{confidence:.2f}%")

        st.info("""
### Clinical Interpretation

The uploaded sclera image shows visual patterns similar to normal eye sclera samples learned by the SSL-enhanced model.
""")

    else:

        confidence = (1 - prediction) * 100

        st.error("⚠️ JAUNDICE DETECTED")

        st.markdown("### 🎯 Confidence Score")
        st.error(f"{confidence:.2f}%")

        st.warning("""
### Clinical Interpretation

The uploaded sclera image contains visual features associated with jaundice samples.

Further clinical evaluation is recommended.
""")

# =========================
# GRAD-CAM SHOWCASE
# =========================

st.markdown("---")

st.subheader("🔥 Explainable AI (Grad-CAM)")

st.write("""
Grad-CAM highlights image regions that contributed most to the model's prediction.

This improves transparency and helps clinicians understand the model's decision-making process.
""")

col1, col2 = st.columns(2)

with col1:

    st.image(
        r"E:\Early_Jaundice_Risk_Prediction_DL\outputs\gradcam\gradcam_jaundice_1.jpg",
        caption="Jaundice Grad-CAM"
    )

with col2:

    st.image(
        r"E:\Early_Jaundice_Risk_Prediction_DL\outputs\gradcam\gradcam_normal_1.jpg",
        caption="Normal Grad-CAM"
    )
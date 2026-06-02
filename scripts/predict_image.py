import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# =========================
# MODEL PATH
# =========================
MODEL_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\models\mobilenet_jaundice_model.h5"

# =========================
# IMAGE PATH
# =========================
IMAGE_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\test_eye.jpg"

# =========================
# PARAMETERS
# =========================
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD IMAGE
# =========================
print("Loading image...")

img = image.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICTION
# =========================
print("Predicting...")

prediction = model.predict(img_array)[0][0]

# =========================
# RESULT
# =========================
if prediction > 0.5:
    print("\nPrediction: NORMAL")
    print(f"Confidence: {prediction * 100:.2f}%")
else:
    print("\nPrediction: JAUNDICE")
    print(f"Confidence: {(1 - prediction) * 100:.2f}%")
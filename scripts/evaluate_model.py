import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Model path
MODEL_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\models\jaundice_model.h5"

# Dataset path
DATASET_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\split"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print(f"\n Test Accuracy: {accuracy * 100:.2f}%")
print(f" Test Loss: {loss:.4f}")

# Predictions
predictions = model.predict(test_generator)

predicted_classes = (predictions > 0.5).astype(int)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# Classification report
print("\n Classification Report:\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))

# Confusion matrix
print("\n Confusion Matrix:\n")

cm = confusion_matrix(true_classes, predicted_classes)

print(cm)
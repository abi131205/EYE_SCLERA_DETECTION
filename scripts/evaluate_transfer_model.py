import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# Model path
MODEL_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\models\mobilenet_jaundice_model.h5"

# Dataset path
DATASET_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\split"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Test generator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_generator)

print(f"\n Test Accuracy: {accuracy * 100:.2f}%")
print(f" Test Loss: {loss:.4f}")

# Predictions
predictions = model.predict(test_generator)

y_true = test_generator.classes

y_pred_probs = predictions[:, 0]

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

# =========================
# SAVE CONFUSION MATRIX
# =========================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Jaundice', 'Normal'],
    yticklabels=['Jaundice', 'Normal']
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

print("Confusion matrix saved.")

# =========================
# ROC CURVE
# =========================

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.savefig("outputs/roc_curve.png")

plt.close()

print("ROC curve saved.")
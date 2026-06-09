# 👁️ Early Jaundice Risk Prediction Using Eye Sclera Images, Self-Supervised Learning and Explainable AI

## 📌 Project Overview

This project presents a non-invasive Artificial Intelligence framework for early jaundice risk prediction using eye sclera images.

The proposed system combines Self-Supervised Learning (SimCLR), MobileNetV2 Transfer Learning, and Grad-CAM Explainable AI to improve prediction performance while maintaining model interpretability.

---

## 👨‍💻 Team Members

### Abijith UK

**Register Number:** E0423021

### Mukesh Kumar

**Register Number:** E0423009

Department of Biomedical Engineering

---

## 🎯 Problem Statement

Traditional jaundice diagnosis generally requires clinical examination and laboratory testing.

This project aims to develop a low-cost, non-invasive AI-based screening system capable of predicting jaundice risk from eye sclera images.

---

## 🚀 Objectives

* Develop an automated jaundice screening framework.
* Analyze eye sclera images using Deep Learning.
* Improve feature learning using Self-Supervised Learning.
* Visualize model decisions using Explainable AI.
* Deploy the system through a Streamlit web application.

---

## 📊 Dataset Information

### Sources

* Zenodo NJN Dataset
* Supporting Eye Sclera Datasets

### Dataset Statistics

| Class    | Images |
| -------- | ------ |
| Normal   | 451    |
| Jaundice | 157    |
| Total    | 608    |

---

## 🛠️ Preprocessing Pipeline

Dataset Collection

⬇️

Data Inspection

⬇️

Data Cleaning

⬇️

Image Resizing

⬇️

Sclera Cropping

⬇️

Data Augmentation

⬇️

Train / Validation / Test Split

---

## 🤖 Baseline Model

### MobileNetV2 Transfer Learning

The initial model was developed using MobileNetV2 as a feature extractor.

### Result

**Accuracy: 67.03%**

---

## 🔬 Self-Supervised Learning Enhancement

### SimCLR

SimCLR was used to learn meaningful image representations before supervised classification.

### SSL Training Results

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.7559 |
| 10    | 0.0457 |

This demonstrates successful representation learning during SSL pretraining.

---

## 🧪 Experimental Results

### Experiment 1

SSL Encoder Trainable

Result:

**Accuracy: 52.86%**

Observation:

Overfitting occurred when the pretrained encoder was fully trainable.

---

### Experiment 2

SSL Encoder Frozen

Result:

**Accuracy: 79.84%**

Observation:

Best generalization performance achieved by freezing the SSL encoder and training only the classifier layers.

---

## 📈 Performance Comparison

| Model                 | Accuracy |
| --------------------- | -------- |
| MobileNetV2 Baseline  | 67.03%   |
| SSL Encoder Trainable | 52.86%   |
| SSL Encoder Frozen    | 79.84%   |

### Improvement

**67.03% → 79.84%**

**Accuracy Gain: +12.81%**

---

## 🔥 Explainable AI

### Grad-CAM

Grad-CAM was implemented to visualize image regions contributing to model predictions.

Features:

* Heatmap Generation
* Attention Visualization
* Model Transparency
* Clinical Interpretability

---

## 💻 Streamlit Deployment

The final model was deployed using Streamlit.

### Features

* Eye Image Upload
* SSL-Based Prediction
* Confidence Score
* Clinical Interpretation
* Grad-CAM Visualization

---

## 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* SimCLR
* Grad-CAM
* OpenCV
* NumPy
* Matplotlib
* Streamlit

---

## 📂 Project Structure

```text
Early_Jaundice_Risk_Prediction_DL/

├── datasets/
├── models/
│   ├── baseline/
│   └── experiments/
├── outputs/
│   ├── gradcam/
│   ├── ssl/
│   └── baseline/
├── scripts/
│   ├── baseline/
│   └── research/
├── test_images/
├── README.md
├── requirements.txt
└── .gitignore
```

## 🌟 Novelty of the Work

The proposed framework integrates:

* Eye Sclera Based Analysis
* SimCLR Self-Supervised Learning
* MobileNetV2 Transfer Learning
* Grad-CAM Explainability
* Streamlit Deployment

This combination improves both performance and interpretability for early jaundice screening.

---

## 🔮 Future Scope

* Larger Multi-Center Dataset
* Advanced Sclera Segmentation
* Mobile Application Deployment
* Real-Time Grad-CAM Integration
* Clinical Validation Studies

---

## 📄 Research Domain

Biomedical Engineering

Artificial Intelligence

Computer Vision

Medical Image Analysis

Explainable AI

Deep Learning

---

## 🙏 Acknowledgement

This project was developed as part of the Biomedical Engineering curriculum and explores the application of modern AI techniques for non-invasive healthcare screening.

---

## 📧 Contact

For academic and research discussions:

Abijith UK (E0423021)

Mukesh Kumar (E0423009)

Department of Biomedical Engineering

```
```

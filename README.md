# 👁️ Early Jaundice Risk Prediction using Eye Sclera Analysis and Deep Learning

## 📌 Overview

This project focuses on early jaundice risk prediction using eye sclera images and deep learning techniques. Traditional jaundice diagnosis often depends on blood tests and clinical observation. This work explores a non-invasive, image-based screening approach using sclera region analysis.

The system performs preprocessing, sclera extraction, dataset preparation, deep learning model training, evaluation, and deployment through a Streamlit interface.

---

## 🎯 Objectives

- Detect jaundice risk using eye sclera images
- Develop a non-invasive screening approach
- Compare CNN and Transfer Learning models
- Evaluate model performance using multiple metrics
- Provide an easy-to-use prediction interface

---

## 🧠 Models Used

### 1. Custom CNN Model
A custom convolutional neural network trained on processed sclera images.

### 2. MobileNet Transfer Learning Model
Transfer learning using MobileNet for improved efficiency and lightweight deployment.

---

## 📂 Project Structure

```text
EARLY_JAUNDICE_RISK_PREDICTION_DL
│
├── preprocessing/
│   ├── augmentation.py
│   ├── clean_dataset.py
│   ├── crop_sclera.py
│   ├── inspect_dataset.py
│   ├── resize_images.py
│   └── split_dataset.py
│
├── scripts/
│   ├── app.py
│   ├── evaluate_model.py
│   ├── evaluate_transfer_model.py
│   ├── predict_image.py
│   ├── train_model.py
│   └── train_transfer_model.py
│
├── models/
│   └── mobilenet_jaundice_model.h5
│
├── outputs/
│   ├── accuracy_graph.png
│   ├── confusion_matrix.png
│   ├── loss_graph.png
│   └── roc_curve.png
│
├── papers/
├── notes/
├── README.md
└── requirements.txt
```

---

## ⚙️ Workflow

### Step 1: Dataset Collection
- Eye sclera image dataset preparation
- Dataset cleaning and inspection

### Step 2: Preprocessing
- Image resizing
- Sclera region cropping
- Image augmentation
- Dataset splitting

### Step 3: Model Training
- CNN training
- MobileNet transfer learning

### Step 4: Evaluation
Performance measured using:

- Accuracy
- Confusion Matrix
- ROC Curve
- Loss Analysis

### Step 5: Deployment
Prediction system deployed using Streamlit.

---

## 📊 Results

The trained models were evaluated using standard classification metrics and visual performance plots.

Included outputs:

- Accuracy Graph
- Loss Graph
- Confusion Matrix
- ROC Curve

---

## 🚀 Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run scripts/app.py
```

---

## 📁 Dataset and Large Files

Due to GitHub storage limitations, datasets and large model files are shared separately through cloud storage.

---

## 🔬 Future Scope

Potential improvements include:

- Grad-CAM based explainability
- Real-time mobile deployment
- Improved sclera segmentation
- Larger and clinically diverse datasets
- Performance comparison with additional architectures

---

## 👥 Team Collaboration

This project supports collaborative development using GitHub and shared cloud storage for large files.

---

## 📚 References

Relevant research papers are included in the `papers/` directory.

---

## ⭐ Project Status

✅ Research and preprocessing completed  
✅ Model development completed  
✅ Evaluation completed  
✅ Streamlit deployment completed  
🔄 Ongoing improvement and experimentation

---

## 📜 License

This project is developed for academic and research purposes.

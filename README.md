# Brain Tumor MRI Classification

## Overview
This project is a deep learning-based system that classifies brain tumors from MRI images. It uses a pretrained DenseNet121 model with transfer learning to identify tumor types.

A Streamlit web application is included, allowing users to upload MRI images and get predictions with confidence scores.

---

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset  
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  

### Details
- Total images: ~7000  
- Image type: MRI scans  
- Format: JPG / PNG  

### Classes
- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

---

## Model
- Model: DenseNet121  
- Technique: Transfer Learning  

The model extracts features from MRI images and classifies them into four tumor categories.

---

## Project Workflow
1. Load dataset  
2. Preprocess images (resize, normalize)  
3. Apply data augmentation  
4. Train DenseNet121 model  
5. Evaluate model performance  
6. Predict tumor type  
7. Deploy using Streamlit  

---

## Project Structure
brain-tumor-classifier/
│
├── app.py
├── requirements.txt
├── models/
│   └── brain_tumor_densenet121.keras
├── notebooks/
│   └── training.ipynb
└── README.md

---

## How to Run

### 1. Clone Repository
git clone https://github.com/HarshaMotupalli/brain-tumor-classifier.git  
cd brain-tumor-classifier  

### 2. Create Virtual Environment
python -m venv env  
env\Scripts\activate  

### 3. Install Requirements
pip install -r requirements.txt  

### 4. Run Application
streamlit run app.py  

---

## Output
- Predicted tumor type  
- Confidence score  
- Probability distribution  

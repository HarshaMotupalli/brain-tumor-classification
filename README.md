# Brain Tumor MRI Classifier

This repository contains a deep learning-based brain tumor classification model and a Streamlit web application to test the model on MRI images.

## Model Details
- Architecture: DenseNet121 (Transfer Learning)  
- Input size: 224 × 224 RGB  
- Classes:
  - Glioma  
  - Meningioma  
  - Pituitary  
  - No Tumor  
- **Accuracy:** ~96% 

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset  
- Link: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle  
- Total images: ~7000+  
- Format: JPG / PNG  

### Structure
- Training folder  
- Testing folder  
- Each class stored in separate directories  

## Project Workflow
1. Load MRI dataset  
2. Preprocess images (resize to 224×224, normalize)  
3. Apply data augmentation  
4. Train DenseNet121 model using transfer learning  
5. Evaluate model performance  
6. Save trained model  
7. Deploy using Streamlit dashboard  
8. Upload image and predict tumor type  

## Project Structure
```bash
brain-tumor-classifier/
│
├── app.py
├── requirements.txt
├── models/
│   └── brain_tumor_densenet121.keras
├── notebooks/
│   └── training.ipynb
└── README.md
```

## Setup Instructions
### 1. Create Virtual Environment
```bash
python -m venv env  
env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
``` 

## Running the Application
```bash
streamlit run app.py
``` 

The application will open in your browser at **http://localhost:8501**

## Output
- Predicted tumor type  
- Confidence score  
- Probability distribution  

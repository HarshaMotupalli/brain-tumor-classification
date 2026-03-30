# # ===============================================
# # Brain Tumor MRI Classification System
# # Model: DenseNet121
# # Interface: Streamlit
# # ===============================================

# # -----------------------------
# # 1. Import Libraries
# # -----------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import preprocess_input


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Dashboard",
    layout="wide"
)

# -----------------------------
# CLEAN CSS (NO BARS)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

/* Remove padding issues */
.block-container {
    padding-top: 2rem;
}

/* Titles */
.title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: white;
}

.section {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    color: #e6edf3;
}

/* Clean cards */
.card {
    background-color: transparent;
    padding: 10px;
}

/* Text */
.small-text {
    color: #9aa4b2;
}

/* Upload area */
.upload-box {
    border: 2px dashed #444;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
}

/* Center image */
.center {
    display: flex;
    justify-content: center;
}

/* Remove ALL unwanted bars */
div[data-testid="stHorizontalBlock"] > div {
    background: none !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Project Details"])


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "C:/brain-tumor-classifier/brain_tumor_densenet121.keras"
    )

model = load_model()


# -----------------------------
# Classes
# -----------------------------
classes = ["glioma", "meningioma", "notumor", "pituitary"]


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image)

    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


# =============================
# DASHBOARD
# =============================
if page == "Dashboard":

    st.markdown("<div class='title'>Brain Tumor MRI Classification</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------
    # Upload Section
    # -----------------------------
    st.markdown("<div class='section'>Upload MRI Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "png", "jpeg"]
    )

    # -----------------------------
    # Show Image
    # -----------------------------
    if uploaded_file:
        image = Image.open(uploaded_file)

        st.markdown("<div class='section'>Preview</div>", unsafe_allow_html=True)

        st.markdown("<div class='center'>", unsafe_allow_html=True)
        st.image(image, width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------
        # Predict Button
        # -----------------------------
        if st.button("Run Prediction"):

            with st.spinner("Analyzing MRI image..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)

            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success("Prediction Completed")

            # -----------------------------
            # Results Section
            # -----------------------------
            st.markdown("<div class='section'>Prediction Results</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.write("Tumor Type")
                st.subheader(predicted_class.upper())

            with col2:
                st.write("Confidence")
                st.subheader(f"{confidence:.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # -----------------------------
            # Graph Section (UPDATED)
            # -----------------------------
            st.markdown("<div class='section'>Probability Distribution</div>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5,3))  # smaller size

            colors = ["#ff4b4b", "#4dabf7", "#51cf66", "#ffd43b"]  # different colors

            bars = ax.bar(classes, prediction[0], color=colors)

            ax.set_ylim([0, 1])
            ax.set_ylabel("Probability")

            # Add values on top
            for i, v in enumerate(prediction[0]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

            plt.xticks(rotation=15)

            st.pyplot(fig)


# =============================
# PROJECT DETAILS (UPDATED)
# =============================
else:

    st.markdown("<div class='title'>Project Overview</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>Introduction</div>", unsafe_allow_html=True)
    st.write("""
This project is a deep learning-based Brain Tumor Classification system designed to automatically 
identify the type of tumor from MRI brain images. It helps in assisting medical professionals by 
providing fast and accurate predictions.
    """)

    st.markdown("<div class='section'>Objective</div>", unsafe_allow_html=True)
    st.write("""
The main objective is to classify MRI images into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The system aims to reduce manual effort and improve diagnostic support.
    """)

    st.markdown("<div class='section'>Dataset</div>", unsafe_allow_html=True)
    st.write("""
The dataset is taken from Kaggle and contains thousands of labeled MRI images.

- Total images: approximately 7000+
- Classes: 4 categories
- Format: JPG/PNG images
- Organized into training and testing folders

Each image represents a brain scan used to train the model.
    """)

    st.markdown("<div class='section'>Model Used</div>", unsafe_allow_html=True)
    st.write("""
DenseNet121 is used as the base model.

It is a deep convolutional neural network that:
- Reuses features from previous layers
- Improves learning efficiency
- Works well for medical image classification

Transfer learning is applied to use pretrained knowledge.
    """)

    st.markdown("<div class='section'>Workflow</div>", unsafe_allow_html=True)
    st.write("""
1. User uploads an MRI image  
2. Image is resized and normalized  
3. Model extracts features using DenseNet  
4. Classification layer predicts tumor type  
5. Output shows prediction and confidence score  
    """)

    st.markdown("<div class='section'>Technologies Used</div>", unsafe_allow_html=True)
    st.write("""
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy and Matplotlib  
    """)

    st.markdown("<div class='section'>Applications</div>", unsafe_allow_html=True)
    st.write("""
- Assisting doctors in diagnosis  
- Medical research  
- Automated screening systems  
    """)

    st.markdown("<div class='section'>Limitations</div>", unsafe_allow_html=True)
    st.write("""
- Depends on dataset quality  
- Requires proper MRI images  
- Not a replacement for medical professionals  
    """)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed for educational and demonstration purposes.| Always consult qualified medical professionals.")

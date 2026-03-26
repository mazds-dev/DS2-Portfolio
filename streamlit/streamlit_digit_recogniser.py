# ============================================================
# INTERACTIVE DIGIT RECOGNITION — Streamlit App
# Student: [Your Name] | Student Number: [Your Number]
# ============================================================
# 
# How to run:
#   1. Install dependencies:
#      pip install streamlit streamlit-drawable-canvas tensorflow numpy pillow
#
#   2. The model file 'digit_recognition_model.keras' is generated
#      by running the notebook first. The app will automatically
#      search for it in the current folder and in ../notebooks/
#
#   3. Run the app:
#      streamlit run streamlit_digit_recogniser.py
#
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras as keras

# --- Page Configuration ---
st.set_page_config(
    page_title="Digit Recogniser",
    page_icon="✍️",
    layout="centered"
)

# --- Load the trained model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    """Load the pre-trained neural network model."""
    import os
    paths = [
        'digit_recognition_model.keras',
        '../notebooks/digit_recognition_model.keras',
    ]
    for path in paths:
        if os.path.exists(path):
            return keras.models.load_model(path)
    st.error("Model not found. Please run the notebook first to generate the model.")
    return None

model = load_model()

# --- App Title ---
st.title("✍️ Handwritten Digit Recogniser")
st.markdown("Draw a digit (0–9) on the canvas below and the neural network will recognise it in real time.")

# --- Try to use streamlit-drawable-canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    
    st.markdown("### Draw here:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=8,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.markdown("### Prediction")
        
        if canvas_result.image_data is not None and model is not None:
            # Get the drawn image
            img_array = canvas_result.image_data
            
            # Check if user has drawn anything (not just black canvas)
            if np.sum(img_array[:, :, :3]) > 50000:
                # Convert to grayscale
                img_gray = np.mean(img_array[:, :, :3], axis=2)
                
                # Convert to PIL image
                img_pil = Image.fromarray(img_gray.astype('uint8'))
                
                # Find bounding box of the drawn digit and crop
                img_array_2d = np.array(img_pil)
                coords = np.argwhere(img_array_2d > 20)
                if len(coords) > 0:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    # Crop to digit with padding
                    padding = 30
                    y0 = max(0, y0 - padding)
                    y1 = min(img_array_2d.shape[0], y1 + padding)
                    x0 = max(0, x0 - padding)
                    x1 = min(img_array_2d.shape[1], x1 + padding)
                    img_cropped = img_pil.crop((x0, y0, x1, y1))
                    
                    # Make it square (MNIST digits are centered in a square)
                    w, h = img_cropped.size
                    max_dim = max(w, h)
                    img_square = Image.new('L', (max_dim, max_dim), 0)
                    offset_x = (max_dim - w) // 2
                    offset_y = (max_dim - h) // 2
                    img_square.paste(img_cropped, (offset_x, offset_y))
                    
                    # Resize to 28x28 (MNIST format)
                    img_resized = img_square.resize((28, 28), Image.LANCZOS)
                else:
                    img_resized = img_pil.resize((28, 28), Image.LANCZOS)
                
                # Normalise to 0-1
                img_normalised = np.array(img_resized).astype('float32') / 255.0
                
                # Flatten for the model (28x28 → 784)
                img_flat = img_normalised.reshape(1, 784)
                
                # Predict
                prediction = model.predict(img_flat, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit] * 100
                
                # Display result
                st.markdown(f"## {predicted_digit}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Show confidence for all digits as a bar chart
                st.markdown("---")
                st.markdown("**All probabilities:**")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                colours = ['#2196F3' if i == predicted_digit else "#e74d3ca6" for i in range(10)]
                ax.bar(range(10), [prediction[0][i] * 100 for i in range(10)], color=colours)
                ax.set_xticks(range(10))
                ax.set_xticklabels([str(i) for i in range(10)])
                ax.set_ylabel('%')
                ax.set_ylim(0, 100)
                st.pyplot(fig)
                plt.close()
                
                # Show what the model sees (28x28 processed image)
                st.markdown("---")
                st.markdown("**What the model sees (28×28):**")
                st.image(img_normalised, width=112, clamp=True)
            else:
                st.markdown("*Draw a digit to see the prediction*")

except ImportError:
    # Fallback: file upload if canvas library not installed
    st.warning("⚠️ The `streamlit-drawable-canvas` library is not installed.")
    st.code("pip install streamlit-drawable-canvas", language="bash")
    
    st.markdown("---")
    st.markdown("### Alternative: Upload an image of a digit")
    
    uploaded_file = st.file_uploader(
        "Upload a digit image (PNG/JPG)", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None and model is not None:
        # Load and process the uploaded image
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img_resized = img.resize((28, 28), Image.LANCZOS)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Your image:**")
            st.image(uploaded_file, width=200)
        
        with col2:
            # Normalise and predict
            img_array = np.array(img_resized).astype('float32') / 255.0
            img_flat = img_array.reshape(1, 784)
            
            prediction = model.predict(img_flat, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = prediction[0][predicted_digit] * 100
            
            st.markdown("### Prediction")
            st.markdown(f"## Digit: {predicted_digit}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
        
        # Show all probabilities
        st.markdown("---")
        st.markdown("**Probability for each digit:**")
        chart_data = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        st.bar_chart(chart_data)


# --- About section ---
with st.expander("ℹ️ About this app"):
    st.markdown("""
    **How it works:**
    1. You draw a digit (0–9) on the canvas
    2. The image is resized to 28×28 pixels (MNIST format)
    3. The pixel values are normalised (0–1 range)
    4. A trained neural network predicts the digit
    5. The model outputs probabilities for all 10 digits
    
    **Model architecture:**
    - Input: 784 neurons (28×28 pixels)
    - Hidden layer 1: 128 neurons (ReLU)
    - Dropout: 20%
    - Hidden layer 2: 64 neurons (ReLU)
    - Dropout: 20%
    - Output: 10 neurons (Softmax)
    
    **Training accuracy:** ~97%+
    
    **Tips for better recognition:**
    - Draw the digit large and centred
    - Use thick strokes
    - Keep the digit simple and clear
    """)

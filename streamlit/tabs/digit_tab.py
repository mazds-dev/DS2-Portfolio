# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# DIGIT RECOGNISER TAB — Neural Network with Interactive Canvas
# Based on the working standalone streamlit_digit_recogniser.py
# ============================================================

import streamlit as st
import numpy as np
import os

from PIL import Image
import tensorflow as tf
from tensorflow import keras


@st.cache_resource
def load_digit_model():
    """Load the trained digit recognition model using absolute paths."""
    # Get the directory where this current script (digit_tab.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define potential locations relative to this script
    # Since models are in the parent folder 'streamlit/', we use '..'
    paths = [
        os.path.join(current_dir, '..', 'digit_recognition_model.keras'), # Parent folder
        os.path.join(current_dir, 'digit_recognition_model.keras'),      # Same folder
        'digit_recognition_model.keras'                                  # Root fallback
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                return keras.models.load_model(path)
            except Exception as e:
                st.error(f"Error loading model at {path}: {e}")
    return None


def render():
    """Render the Digit Recogniser tab."""
    st.header("✍️ Handwritten Digit Recogniser")
    st.markdown(
        "Draw a digit (0–9) on the canvas below and the neural network "
        "will recognise it in real time."
    )

    model = load_digit_model()

    if model is None:
        st.error(
            "Model not found. Please run the notebook "
            "`03_Neural_Network_Digit_Recognition.ipynb` to generate "
            "`digit_recognition_model.keras`."
        )
        return

    # --- Try to import drawable canvas ---
    try:
        from streamlit_drawable_canvas import st_canvas

        st.markdown("### Draw here:")

        col1, col2 = st.columns([2, 1])

        with col1:
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=8,
                stroke_color="white",
                background_color="black",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="digit_canvas",
            )

        with col2:
            st.markdown("### Prediction")

            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data

                # Only run prediction if user actually drew something
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
                    predicted_digit = int(np.argmax(prediction))
                    confidence = prediction[0][predicted_digit] * 100

                    # Display result
                    st.markdown(f"## {predicted_digit}")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")

                    # Probability chart
                    st.markdown("---")
                    st.markdown("**All probabilities:**")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 3))
                    colours = [
                        '#2196F3' if i == predicted_digit else '#e74c3c'
                        for i in range(10)
                    ]
                    ax.bar(
                        range(10),
                        [prediction[0][i] * 100 for i in range(10)],
                        color=colours,
                    )
                    ax.set_xticks(range(10))
                    ax.set_xticklabels([str(i) for i in range(10)], fontsize=10)
                    ax.set_ylabel('%')
                    ax.set_ylim(0, 100)
                    st.pyplot(fig, use_container_width=False)
                    plt.close()

                    # Show what the model sees
                    st.markdown("---")
                    st.markdown("**What the model sees (28×28):**")
                    st.image(img_normalised, width=112, clamp=True)
                else:
                    st.markdown("_Draw a digit to see the prediction_")

    except ImportError:
        # Fallback: file upload if canvas library not installed
        st.warning(
            "⚠️ The `streamlit-drawable-canvas` library is not installed."
        )
        st.code("pip install streamlit-drawable-canvas", language="bash")

        st.markdown("---")
        st.markdown("### Alternative: Upload an image of a digit")

        uploaded_file = st.file_uploader(
            "Upload a digit image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('L')
            img_resized = img.resize((28, 28), Image.LANCZOS)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Your image:**")
                st.image(uploaded_file, width=200)

            with col2:
                img_arr = np.array(img_resized).astype('float32') / 255.0
                img_flat = img_arr.reshape(1, 784)

                prediction = model.predict(img_flat, verbose=0)
                predicted_digit = int(np.argmax(prediction))
                confidence = prediction[0][predicted_digit] * 100

                st.markdown("### Prediction")
                st.markdown(f"## Digit: {predicted_digit}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")

    # --- About section ---
    with st.expander("ℹ️ How digit recognition works"):
        st.markdown(
            """
            **The full pipeline:**
            1. A digit is drawn on the canvas
            2. The image is resized to 28×28 pixels (MNIST format)
            3. The pixel values are normalised (0–1 range)
            4. The neural network processes the image and outputs
               probabilities for each digit
            5. The highest probability is the final prediction

            **Model architecture:**
            - Input: 784 neurons (28×28 pixels)
            - Hidden layer 1: 128 neurons (ReLU)
            - Dropout: 20%
            - Hidden layer 2: 64 neurons (ReLU)
            - Dropout: 20%
            - Output: 10 neurons (Softmax)

            **Tips for better recognition:**
            - Draw the digit large and centred
            - Use a single continuous stroke where possible
            - Keep the digit simple and clear
            """
        )

    with st.expander("💻 Source Code — ANN Architecture & Training"):
        st.code("""# Build the ANN
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Normalise and flatten images
X_train_flat = X_train.astype('float32') / 255.0
X_train_flat = X_train_flat.reshape(-1, 784)

# Train for 20 epochs
history = model.fit(
    X_train_flat, y_train_cat,
    epochs=20, batch_size=128, validation_split=0.2
)
# Test accuracy: ~97%""", language="python")
# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# FASHION CLASSIFIER TAB — CNN Product Classification
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import tensorflow as tf
from tensorflow import keras


CATEGORY_NAMES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


@st.cache_resource
def load_fashion_model():
    paths = [
        "fashion_cnn_model.keras",
        "../fashion_cnn_model.keras",
        "models/fashion_cnn_model.keras",
        "../models/fashion_cnn_model.keras",
        "../notebooks/fashion_cnn_model.keras",
    ]
    for path in paths:
        if os.path.exists(path):
            return keras.models.load_model(path)
    return None


@st.cache_data
def load_fashion_test_data():
    (_, _), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return X_test, y_test


def render():
    st.header("👕 Fashion Item Classifier")
    st.markdown(
        "Automatically categorises fashion items using a **Convolutional "
        "Neural Network (CNN)**. Trained on Fashion MNIST with ~90% "
        "accuracy. Simulates an e-commerce product cataloguing system."
    )

    model = load_fashion_model()

    if model is None:
        st.error(
            "Fashion CNN model not found. Run "
            "`05_CNN_Fashion_Classification.ipynb` to generate "
            "`fashion_cnn_model.keras`."
        )
        return

    st.subheader("Choose input method")
    mode = st.radio(
        "How would you like to classify an item?",
        options=["🎲 Random sample from dataset", "📤 Upload your own image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("🎲"):
        _random_sample_mode(model)
    else:
        _upload_mode(model)

    st.markdown("---")

    with st.expander("ℹ️ How CNN is different from ANN"):
        st.markdown(
            """
            The **Digit Recogniser** tab uses a basic ANN — it treats the
            image as a flat list of 784 pixels, losing all spatial
            information.

            The **CNN** here is much more powerful for image data because it:
            - Uses **convolutional filters** that slide over the image
              detecting local patterns (edges, corners, textures)
            - Uses **pooling** to reduce image size while keeping important
              features
            - Builds a **hierarchy** — early layers detect simple patterns
              (edges), later layers combine them into complex shapes

            **Why Fashion MNIST is harder than digit MNIST:**
            - Clothing items have more visual variation
            - Some categories are visually similar (Shirt, T-shirt, Pullover)
            - The CNN achieves ~90% accuracy vs the ANN's ~87%
            """
        )

    with st.expander("💻 Source Code — CNN Architecture & Training"):
        st.code("""# Build the CNN
cnn_model = models.Sequential([
    # First convolutional block — detects edges
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block — detects shapes
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten 2D feature maps to 1D
    layers.Flatten(),

    # Dense classifier
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train for 10 epochs
history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    epochs=10, batch_size=128, validation_split=0.2
)
# Test accuracy: ~90.3%""", language="python")


def _random_sample_mode(model):
    X_test, y_test = load_fashion_test_data()

    if "fashion_sample_idx" not in st.session_state:
        st.session_state.fashion_sample_idx = np.random.randint(0, len(X_test))

    if st.button("🎲 Show random item", use_container_width=True):
        st.session_state.fashion_sample_idx = np.random.randint(0, len(X_test))

    idx = st.session_state.fashion_sample_idx
    img = X_test[idx]
    actual_category = CATEGORY_NAMES[y_test[idx]]

    # Display image
    st.markdown(f"**Actual category:** {actual_category}")

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)
    plt.close()

    # Predict
    img_norm = img.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 28, 28, 1)
    prediction = model.predict(img_input, verbose=0)
    predicted_idx = int(np.argmax(prediction))
    predicted_category = CATEGORY_NAMES[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100

    # Result
    if predicted_category == actual_category:
        st.success(f"### ✅ Prediction: **{predicted_category}**")
    else:
        st.error(f"### ❌ Prediction: **{predicted_category}**")

    st.markdown(f"**Confidence:** {confidence:.1f}%")

    # Probability chart
    st.markdown("**All probabilities:**")
    fig, ax = plt.subplots(figsize=(5, 2.4))
    colours = [
        "#2ecc71" if (i == predicted_idx and i == y_test[idx])
        else "#e74c3c" if i == predicted_idx
        else "#bdc3c7"
        for i in range(10)
    ]
    probs = prediction[0] * 100
    ax.bar(range(10), probs, color=colours)
    ax.set_xticks(range(10))
    ax.set_xticklabels([n[:6] for n in CATEGORY_NAMES], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    st.pyplot(fig, use_container_width=False)
    plt.close()


def _upload_mode(model):
    uploaded = st.file_uploader(
        "Upload a clothing image (converted to 28×28 grayscale)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        st.info(
            "💡 Upload a photo of a clothing item or accessory. For best "
            "results, use a clean background and a single item centred."
        )
        return

    img_original = Image.open(uploaded).convert("L")

    st.markdown("**Your image:**")
    st.image(uploaded, width=250)

    img_resized = img_original.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_resized)
    if img_array.mean() > 128:
        img_array = 255 - img_array

    st.markdown("**Model input (28×28, inverted if needed):**")
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(img_array, cmap="gray")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)
    plt.close()

    img_norm = img_array.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 28, 28, 1)
    prediction = model.predict(img_input, verbose=0)
    predicted_idx = int(np.argmax(prediction))
    predicted_category = CATEGORY_NAMES[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100

    st.markdown(f"### 🏷️ **{predicted_category}**")
    st.markdown(f"**Confidence:** {confidence:.1f}%")

    st.markdown("**All probabilities:**")
    fig, ax = plt.subplots(figsize=(5, 2.4))
    colours = [
        "#2196F3" if i == predicted_idx else "#bdc3c7" for i in range(10)
    ]
    probs = prediction[0] * 100
    ax.bar(range(10), probs, color=colours)
    ax.set_xticks(range(10))
    ax.set_xticklabels([n[:6] for n in CATEGORY_NAMES], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    st.pyplot(fig, use_container_width=False)
    plt.close()

    st.caption(
        "Note: The model was trained on simple grayscale images of isolated "
        "items. Real-world photos with complex backgrounds may give less "
        "accurate results."
    )
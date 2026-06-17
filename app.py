import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model("mobilenet_cats_dogs.keras")

st.title("🐱 Cats vs Dogs Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert RGBA/PNG images to RGB
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    img = img.resize((224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    prediction = model.predict(
        img_array,
        verbose=0
    )

    confidence = float(prediction[0][0])

    st.write(f"Raw Prediction: {confidence:.6f}")

    if confidence > 0.5:
        st.success(
            f"🐶 Dog ({confidence * 100:.2f}% confidence)"
        )
    else:
        st.success(
            f"🐱 Cat ({(1 - confidence) * 100:.2f}% confidence)"
        )
        
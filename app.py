import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

from model_comparison import get_model_results
from gradcam import make_gradcam_heatmap
from gradcam import overlay_heatmap

# -------------------------
# Load Model
# -------------------------

model = tf.keras.models.load_model(
    "mobilenet_cats_dogs.keras"
)

# -------------------------
# App Title
# -------------------------

st.title("🐱 Cats vs Dogs Classifier")

st.write(
    "Upload an image or use your webcam and the model will predict whether it is a cat or a dog."
)

# -------------------------
# Image Input
# -------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input(
    "Or take a picture using your webcam"
)

image_source = uploaded_file if uploaded_file else camera_image

# -------------------------
# Prediction Section
# -------------------------

if image_source is not None:

    img = Image.open(image_source).convert("RGB")

    st.image(
        img,
        caption="Input Image",
        use_container_width=True
    )

    original_img = np.array(img)

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    confidence = float(prediction[0][0])

    if confidence > 0.5:

        label = "Dog"
        confidence_percent = confidence * 100

    else:

        label = "Cat"
        confidence_percent = (1 - confidence) * 100

    st.subheader("Prediction Result")

    col1, col2 = st.columns([1, 1])

    # -------------------------
    # Prediction Text
    # -------------------------

    with col1:

        if label == "Dog":

            st.success(
                f"🐶 Dog ({confidence_percent:.2f}% confidence)"
            )

        else:

            st.success(
                f"🐱 Cat ({confidence_percent:.2f}% confidence)"
            )

        st.progress(
            confidence_percent / 100
        )

        st.write(
            f"Raw Prediction Score: {confidence:.6f}"
        )

    # -------------------------
    # Gauge Chart
    # -------------------------

    with col2:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence_percent,
                title={
                    'text': "Prediction Confidence"
                },
                gauge={
                    'axis': {
                        'range': [0, 100]
                    },
                    'bar': {
                        'thickness': 0.3
                    },
                    'steps': [
                        {
                            'range': [0, 50],
                            'color': "lightgray"
                        },
                        {
                            'range': [50, 80],
                            'color': "gray"
                        },
                        {
                            'range': [80, 100],
                            'color': "darkgray"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------
    # Grad-CAM
    # -------------------------

    try:

        heatmap = make_gradcam_heatmap(
            img_array,
            model
        )

        gradcam_img = overlay_heatmap(
            heatmap,
            original_img
        )

        st.subheader(
            "🔥 Explainable AI (Grad-CAM)"
        )

        st.image(
            gradcam_img,
            caption="Highlighted regions used by the model",
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

    # -------------------------
    # Model Comparison Dashboard
    # -------------------------

    st.divider()

    st.header(
        "📊 Model Comparison Dashboard"
    )

    comparison_df = get_model_results()

    st.subheader(
        "Model Performance"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader(
        "Accuracy Comparison"
    )

    st.bar_chart(
        comparison_df.set_index(
            "Model"
        )["Accuracy"]
    )

    st.subheader(
        "Model Size Comparison"
    )

    st.bar_chart(
        comparison_df.set_index(
            "Model"
        )["Parameters"]
    )

    st.info(
        """
        MobileNetV2 achieved higher accuracy while using
        transfer learning from ImageNet. This allows the
        model to learn robust image features and outperform
        a custom CNN model on the Cats vs Dogs dataset.
        """
    )

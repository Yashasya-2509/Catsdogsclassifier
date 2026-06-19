import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("mobilenetv2_1.00_224")

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("out_relu").output,
            base_model.output
        ]
    )

    conv_outputs, predictions = grad_model(img_array)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(
    heatmap,
    original_img
):

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original_img.shape[1],
            original_img.shape[0]
        )
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    result = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return result
    

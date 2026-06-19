import tensorflow as tf

model = tf.keras.models.load_model(
    "mobilenet_cats_dogs.keras"
)

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

print("Base model found")

last_conv_layer = base_model.get_layer(
    "out_relu"
)

print("Last conv layer:", last_conv_layer.name)

print("Output shape:", last_conv_layer.output.shape)

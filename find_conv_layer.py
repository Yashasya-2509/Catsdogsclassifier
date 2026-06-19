import tensorflow as tf

model = tf.keras.models.load_model(
    "mobilenet_cats_dogs.keras"
)

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

for layer in reversed(base_model.layers):
    if len(layer.output.shape) == 4:
        print("Last Conv Layer:", layer.name)
        break

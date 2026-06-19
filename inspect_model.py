import tensorflow as tf

model = tf.keras.models.load_model(
    "mobilenet_cats_dogs.keras"
)

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

print(type(base_model))

for i, layer in enumerate(base_model.layers[-20:]):
    print(layer.name)  

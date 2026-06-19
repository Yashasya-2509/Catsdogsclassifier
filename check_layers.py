import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model("mobilenet_cats_dogs.keras")

print("Model loaded")

for layer in model.layers:
    print(layer.name)

print("Done")

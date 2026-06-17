import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model("mobilenet_cats_dogs.keras")

img_path = sys.argv[1]

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

# DO NOT divide by 255 here
# The model already contains a Rescaling layer

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array, verbose=0)

confidence = float(prediction[0][0])

print(f"Raw Prediction: {confidence:.6f}")

if confidence > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("model/cat_dog_cnn_best.h5")

# Image path (change this)
img_path = "dataset/data/test/2742.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")

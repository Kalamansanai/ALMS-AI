import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('model.h5')

def detect_image(model, image):
    prediction = model.predict(np.array([image]))[0]
    prediction = np.array([1 if x > 0.5 else 0 for x in prediction])
    return prediction
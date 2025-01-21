import tensorflow as tf
import cv2
import numpy as np

def detect_image(image):
    return model.predict(np.array([image]))[0]

def preprocess_image(img):
    img = img[110:712 ,180:1000]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= cv2.resize(img, (224, 224))
    _, img = cv2.threshold(img, 90 , 255, cv2.THRESH_BINARY)
    img = img / 255.0
    return img    

model = tf.keras.models.load_model('model.h5')

# Initialize the camera (0 is the default camera index)
camera = cv2.VideoCapture(0)

# Set resolution to 1280x720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Camera stream of images
while True:
    ret, frame = camera.read()

    if ret:
        # Preprocess the image
        frame = preprocess_image(frame)
        # Detect the image
        prediction = detect_image(frame)
        prediction = [1 if x > 0.5 else 0 for x in prediction]
        prediction = np.array(prediction)

        print(prediction)




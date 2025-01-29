import cv2

def setup_camera():
    # Initialize the camera (0 is the default camera index)
    camera = cv2.VideoCapture(0)
    # Set resolution to 1280x720
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def preprocess_image(img):
    img = img[110:712 ,180:1000]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= cv2.resize(img, (224, 224))
    _, img = cv2.threshold(img, 90 , 255, cv2.THRESH_BINARY)
    img = img / 255.0
    return img   
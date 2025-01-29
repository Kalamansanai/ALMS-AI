import lib.camera as camera
import lib.detect_hands as hands
import lib.detect_parts as parts

# Setup the camera and model
camera = camera.setup_camera()
model = parts.load_model()
hands = hands.setup_hands()

while camera.isOpened():

    ret, frame = camera.read()

    if ret:
        # Detect the image for hands
        hand_detected = hands.detect_hands(hands, frame)
        # Look for change in the image
        if hand_detected:
            # Preprocess the image
            frame = camera.preprocess_image(frame)
            prediction = parts.detect_image(frame)


camera.release()
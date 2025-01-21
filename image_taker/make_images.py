import cv2
import time

# Initialize the camera (0 is the default camera index)
camera = cv2.VideoCapture(0)

# Set resolution to 1280x720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

with open("image_number.txt", "r") as f:
    image_number = int(f.read())

try:
    while True:
        # Wait for user input to capture the next frame
        input("Press Enter to capture another image...")

        # Clear the buffer by reading multiple frames
        for _ in range(5):
            camera.read()
            time.sleep(0.1)  # Small delay to allow the camera to refresh

        # Capture the final frame
        ret, frame = camera.read()

        if ret:
            # Save the captured image
            image_path = f'/home/pi/image_gen/images/image_{image_number}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved at {image_path}"),
            image_number += 1   
        else:
            print("Error: Unable to capture image.")

except KeyboardInterrupt:
    print("\nExiting program.")

finally:
    # Release the camera
    camera.release()
    cv2.destroyAllWindows()

    # Update the image number for the next run
    with open("image_number.txt", "w") as f:
        f.write(str(image_number))
        print(f"Next image number: {image_number}")
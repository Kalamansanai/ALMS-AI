import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,  # For video input
    max_num_hands=2,         # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        result = hands.process(rgb_frame)

        # Draw landmarks and connections
        if result.multi_hand_landmarks:
            print("Hand detected")

        else:
            print("No hand detected")

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()


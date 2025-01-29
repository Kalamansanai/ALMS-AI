import mediapipe as mp
import cv2

def setup_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # For video input
        max_num_hands=2,         # Detect up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    return hands

def detect_hands(hands, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    result = hands.process(rgb_frame)

    # Draw landmarks and connections
    if result.multi_hand_landmarks:
        print("Hand detected")
        return True

    
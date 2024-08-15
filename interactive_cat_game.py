import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained cat detection model
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Game variables
score = 0
cat_toy = None
cat_detected = False


def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_count = 0

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        finger_count += 1

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count


def create_cat_toy(frame_shape):
    return (random.randint(50, frame_shape[1] - 50), random.randint(50, frame_shape[0] - 50))


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect hands
    hand_results = hands.process(rgb_image)

    # Detect cats
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process hand landmarks
    finger_count = 0
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

    # Process cat detection
    if len(cats) > 0:
        cat_detected = True
        (x, y, w, h) = max(cats, key=lambda x: x[2] * x[3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw cat features
        cv2.line(image, (x, y), (x + w // 2, y - h // 2), (0, 255, 0), 2)
        cv2.line(image, (x + w, y), (x + w // 2, y - h // 2), (0, 255, 0), 2)
        whisker_start = (x + w // 2, y + h // 2)
        cv2.line(image, whisker_start, (x, y + h // 2), (0, 255, 0), 1)
        cv2.line(image, whisker_start, (x + w, y + h // 2), (0, 255, 0), 1)
    else:
        cat_detected = False

    # Game logic
    if cat_toy is None or not cat_detected:
        cat_toy = create_cat_toy(image.shape)

    cv2.circle(image, cat_toy, 20, (0, 0, 255), -1)

    if cat_detected:
        cat_center = (x + w // 2, y + h // 2)
        distance = np.sqrt((cat_center[0] - cat_toy[0]) ** 2 + (cat_center[1] - cat_toy[1]) ** 2)

        if distance < 50:  # Cat caught the toy
            score += finger_count + 1
            cat_toy = create_cat_toy(image.shape)

    # Display game information
    cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Fingers: {finger_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if cat_detected:
        cv2.putText(image, "Cat detected!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No cat detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Cat and Finger Game', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

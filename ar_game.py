import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game variables
score = 0
lives = 3
game_over = False
last_shot_time = 0
shot_cooldown = 0.5  # seconds

# Spaceship
spaceship_width = 60
spaceship_height = 80


# Asteroids
class Asteroid:
    def __init__(self):
        self.x = random.randint(0, frame_width)
        self.y = 0
        self.speed = random.randint(2, 5)
        self.size = random.randint(20, 50)

    def move(self):
        self.y += self.speed

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), self.size, (0, 165, 255), -1)


asteroids = []


# Bullets
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 10

    def move(self):
        self.y -= self.speed

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), 5, (0, 255, 0), -1)


bullets = []


def draw_spaceship(frame, x, y):
    points = np.array([
        [x, y + spaceship_height],
        [x + spaceship_width // 2, y],
        [x + spaceship_width, y + spaceship_height]
    ], np.int32)
    cv2.fillPoly(frame, [points], (0, 255, 255))
    cv2.line(frame, (x + spaceship_width // 2, y), (x + spaceship_width // 2, y + spaceship_height), (0, 0, 255), 2)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_image)

    if not game_over:
        # Spawn new asteroids
        if random.random() < 0.02:
            asteroids.append(Asteroid())

        # Move and draw bullets
        for bullet in bullets[:]:
            bullet.move()
            bullet.draw(image)
            if bullet.y < 0:
                bullets.remove(bullet)

        # Move and draw asteroids, check for collisions
        for asteroid in asteroids[:]:
            asteroid.move()
            asteroid.draw(image)

            # Check for collision with bullets
            for bullet in bullets[:]:
                if (bullet.x - asteroid.x) ** 2 + (bullet.y - asteroid.y) ** 2 < asteroid.size ** 2:
                    asteroids.remove(asteroid)
                    bullets.remove(bullet)
                    score += 1
                    break

            # Check if asteroid is off-screen or collides with spaceship
            if asteroid.y > frame_height:
                asteroids.remove(asteroid)
                lives -= 1
                if lives <= 0:
                    game_over = True

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the position of the index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * frame_width)
                y = int(index_finger_tip.y * frame_height)

                # Draw spaceship at index finger tip position
                draw_spaceship(image, x - spaceship_width // 2, y - spaceship_height // 2)

                # Shoot bullet if thumb is up
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                if thumb_tip.y < thumb_ip.y and time.time() - last_shot_time > shot_cooldown:
                    bullets.append(Bullet(x, y - spaceship_height // 2))
                    last_shot_time = time.time()

    # Display game information
    cv2.putText(image, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Lives: {lives}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        cv2.putText(image, "GAME OVER", (frame_width // 2 - 100, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(image, f"Final Score: {score}", (frame_width // 2 - 100, frame_height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Press 'R' to Restart", (frame_width // 2 - 120, frame_height // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('AR Space Shooter', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and game_over:
        # Reset game
        score = 0
        lives = 3
        game_over = False
        asteroids.clear()
        bullets.clear()

cap.release()
cv2.destroyAllWindows()

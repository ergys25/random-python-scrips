import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import csv
import os
import json
import pickle
import matplotlib.pyplot as plt
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AdvancedSmartHomeAssistant:
    def __init__(self):
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.last_notification_time = 0
        self.notification_cooldown = 5  # seconds
        self.gesture_history = deque(maxlen=10)
        self.posture_history = deque(maxlen=100)
        self.blink_history = deque(maxlen=100)
        self.activity_history = deque(maxlen=100)

        self.data_dir = "smart_home_data"
        os.makedirs(self.data_dir, exist_ok=True)

        self.posture_model, self.scaler = self.load_or_train_posture_model()
        self.activity_threshold = 0.1
        self.blink_threshold = 0.3

        self.room_temperature = 22  # simulated room temperature in Celsius
        self.room_humidity = 50  # simulated room humidity in percentage

        self.daily_water_intake = 0
        self.daily_step_count = 0
        self.last_step_time = time.time()

        self.task_list = self.load_tasks()
        self.focus_mode = False
        self.focus_start_time = None

        self.emotion_history = deque(maxlen=100)

        self.state_file = os.path.join(self.data_dir, "assistant_state.pkl")
        self.load_state()

    def save_state(self):
        state = {
            'posture_history': self.posture_history,
            'blink_history': self.blink_history,
            'activity_history': self.activity_history,
            'daily_water_intake': self.daily_water_intake,
            'daily_step_count': self.daily_step_count,
            'last_step_time': self.last_step_time,
            'focus_mode': self.focus_mode,
            'focus_start_time': self.focus_start_time,
            'emotion_history': self.emotion_history,
            'room_temperature': self.room_temperature,
            'room_humidity': self.room_humidity
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            self.posture_history = state['posture_history']
            self.blink_history = state['blink_history']
            self.activity_history = state['activity_history']
            self.daily_water_intake = state['daily_water_intake']
            self.daily_step_count = state['daily_step_count']
            self.last_step_time = state['last_step_time']
            self.focus_mode = state['focus_mode']
            self.focus_start_time = state['focus_start_time']
            self.emotion_history = state['emotion_history']
            self.room_temperature = state['room_temperature']
            self.room_humidity = state['room_humidity']

    def load_or_train_posture_model(self):
        if os.path.exists(os.path.join(self.data_dir, "posture_data.csv")):
            data = np.genfromtxt(os.path.join(self.data_dir, "posture_data.csv"), delimiter=',')
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(data_scaled)
            return model, scaler
        else:
            return None, None

    def save_posture_data(self, landmarks):
        data = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
        with open(os.path.join(self.data_dir, "posture_data.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def detect_poor_posture(self, landmarks):
        if self.posture_model is None or self.scaler is None:
            # Simplified posture detection if no model is available
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y

            return left_ear > left_shoulder and right_ear > right_shoulder
        else:
            data = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
            try:
                data_scaled = self.scaler.transform([data])
                return self.posture_model.predict(data_scaled)[0] == -1
            except Exception as e:
                print(f"Error in posture detection: {str(e)}")
                return False  # Assume good posture in case of error

    def recognize_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        index_middle_distance = np.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)

        if thumb_index_distance < 0.05:
            return "PINCH"
        elif index_middle_distance < 0.05:
            return "PEACE"
        elif thumb_tip.y < index_tip.y and index_tip.y < middle_tip.y:
            return "THUMBS_UP"
        else:
            return "NONE"

    def control_device(self, gesture):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) >= 3:
            if all(g == "PINCH" for g in list(self.gesture_history)[-3:]):
                return "TOGGLE_LIGHT"
            elif all(g == "PEACE" for g in list(self.gesture_history)[-3:]):
                return "ADJUST_THERMOSTAT"
            elif all(g == "THUMBS_UP" for g in list(self.gesture_history)[-3:]):
                return "PLAY_MUSIC"
        return None

    def detect_activity(self, landmarks):
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        wrist_movement = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        ankle_movement = np.sqrt((left_ankle.x - right_ankle.x) ** 2 + (left_ankle.y - right_ankle.y) ** 2)

        if wrist_movement > self.activity_threshold and ankle_movement < self.activity_threshold:
            return "WORKING"
        elif ankle_movement > self.activity_threshold:
            return "WALKING"
        else:
            return "IDLE"

    def detect_blink(self, face_landmarks):
        left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[160], face_landmarks.landmark[158],
                    face_landmarks.landmark[133]]
        right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[385], face_landmarks.landmark[387],
                     face_landmarks.landmark[263]]

        left_eye_ratio = self.get_eye_aspect_ratio(left_eye)
        right_eye_ratio = self.get_eye_aspect_ratio(right_eye)

        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        return eye_ratio < self.blink_threshold

    def get_eye_aspect_ratio(self, eye):
        vertical_dist1 = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[3].x, eye[3].y]))
        vertical_dist2 = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[0].x, eye[0].y]))
        horizontal_dist = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist)

    def update_environmental_data(self):
        # Simulate changes in room temperature and humidity
        self.room_temperature += np.random.uniform(-0.5, 0.5)
        self.room_humidity += np.random.uniform(-1, 1)
        self.room_humidity = max(0, min(100, self.room_humidity))

    def update_health_data(self, activity):
        current_time = time.time()
        if activity == "WALKING" and current_time - self.last_step_time > 0.5:
            self.daily_step_count += 1
            self.last_step_time = current_time

        # Simulate water intake
        if np.random.random() < 0.01:  # 1% chance of drinking water every frame
            self.daily_water_intake += 250  # ml

    def load_tasks(self):
        try:
            with open(os.path.join(self.data_dir, "tasks.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_tasks(self):
        with open(os.path.join(self.data_dir, "tasks.json"), "w") as f:
            json.dump(self.task_list, f)

    def add_task(self, task):
        self.task_list.append({"task": task, "completed": False})
        self.save_tasks()

    def complete_task(self, index):
        if 0 <= index < len(self.task_list):
            self.task_list[index]["completed"] = True
            self.save_tasks()

    def toggle_focus_mode(self):
        self.focus_mode = not self.focus_mode
        if self.focus_mode:
            self.focus_start_time = time.time()
        else:
            self.focus_start_time = None

    def get_focus_duration(self):
        if self.focus_mode and self.focus_start_time:
            return time.time() - self.focus_start_time
        return 0

    def detect_emotion(self, face_landmarks):
        # Simplified emotion detection based on eyebrow and mouth positions
        left_eyebrow = face_landmarks.landmark[70].y
        right_eyebrow = face_landmarks.landmark[300].y
        left_mouth = face_landmarks.landmark[61].y
        right_mouth = face_landmarks.landmark[291].y

        eyebrow_position = (left_eyebrow + right_eyebrow) / 2
        mouth_position = (left_mouth + right_mouth) / 2

        if eyebrow_position < 0.35 and mouth_position > 0.6:
            return "HAPPY"
        elif eyebrow_position > 0.4 and mouth_position < 0.5:
            return "SAD"
        elif eyebrow_position < 0.35 and mouth_position < 0.5:
            return "ANGRY"
        else:
            return "NEUTRAL"

    def generate_daily_report(self):
        plt.figure(figsize=(15, 10))

        # Posture plot
        plt.subplot(2, 2, 1)
        posture_data = [1 if p else 0 for p in self.posture_history]
        plt.plot(posture_data)
        plt.title("Posture Over Time")
        plt.ylabel("Good Posture (1) / Bad Posture (0)")

        # Activity plot
        plt.subplot(2, 2, 2)
        activity_data = [1 if a == "WORKING" else 0.5 if a == "WALKING" else 0 for a in self.activity_history]
        plt.plot(activity_data)
        plt.title("Activity Over Time")
        plt.ylabel("Working (1) / Walking (0.5) / Idle (0)")

        # Blink rate plot
        plt.subplot(2, 2, 3)
        blink_data = [1 if b else 0 for b in self.blink_history]
        plt.plot(blink_data)
        plt.title("Blink Rate Over Time")
        plt.ylabel("Blink (1) / No Blink (0)")

        # Emotion plot
        plt.subplot(2, 2, 4)
        emotion_data = [1 if e == "HAPPY" else 0.66 if e == "NEUTRAL" else 0.33 if e == "SAD" else 0 for e in
                        self.emotion_history]
        plt.plot(emotion_data)
        plt.title("Emotion Over Time")
        plt.ylabel("Happy (1) / Neutral (0.66) / Sad (0.33) / Angry (0)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, f"daily_report_{datetime.date.today()}.png"))
        plt.close()

    def run(self):
        print("Entering run method...")
        self.cap = cv2.VideoCapture(0)
        try:
            if not self.cap.isOpened():
                print("Error: Unable to open webcam.")
                return

            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_results = self.pose.process(image)
                hand_results = self.hands.process(image)
                face_results = self.face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Initialize status messages
                posture_status = "Posture: Good"
                gesture_status = "Gesture: None"
                activity_status = "Activity: Unknown"
                blink_status = "Blink: Not detected"
                emotion_status = "Emotion: Unknown"

                if pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    landmarks = pose_results.pose_landmarks.landmark

                    # Posture detection
                    poor_posture = self.detect_poor_posture(landmarks)
                    if poor_posture:
                        posture_status = "Posture: Poor - Please correct"
                    self.posture_history.append(not poor_posture)
                    self.save_posture_data(landmarks)

                    # Activity detection
                    activity = self.detect_activity(landmarks)
                    activity_status = f"Activity: {activity}"
                    self.activity_history.append(activity)
                    self.update_health_data(activity)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        gesture = self.recognize_gesture(hand_landmarks)
                        device_action = self.control_device(gesture)
                        if device_action:
                            gesture_status = f"Gesture: {device_action}"

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

                        # Blink detection
                        blink_detected = self.detect_blink(face_landmarks)
                        blink_status = "Blink: Detected" if blink_detected else "Blink: Not detected"
                        self.blink_history.append(blink_detected)

                        # Emotion detection
                        emotion = self.detect_emotion(face_landmarks)
                        emotion_status = f"Emotion: {emotion}"
                        self.emotion_history.append(emotion)

                # Update environmental data
                self.update_environmental_data()

                # Display status messages on the image
                cv2.putText(image, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, gesture_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, activity_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, blink_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, emotion_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display environmental data
                cv2.putText(image, f"Temperature: {self.room_temperature:.1f}°C", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Humidity: {self.room_humidity:.1f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

                # Display health data
                cv2.putText(image, f"Steps: {self.daily_step_count}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(image, f"Water: {self.daily_water_intake}ml", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

                # Display focus mode status
                focus_status = "Focus Mode: ON" if self.focus_mode else "Focus Mode: OFF"
                cv2.putText(image, focus_status, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if self.focus_mode:
                    focus_duration = self.get_focus_duration()
                    cv2.putText(image, f"Focus Time: {focus_duration:.0f}s", (10, 330), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 165, 255), 2)

                # Display task list
                task_y = 360
                for i, task in enumerate(self.task_list[:5]):  # Display up to 5 tasks
                    status = "✓" if task["completed"] else "□"
                    cv2.putText(image, f"{status} {task['task']}", (10, task_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 200), 2)
                    task_y += 30

                cv2.imshow('Advanced Smart Home Assistant', image)

                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('f'):  # Toggle focus mode
                    self.toggle_focus_mode()
                elif key == ord('t'):  # Add a new task
                    task = input("Enter a new task: ")
                    self.add_task(task)
                elif key == ord('c'):  # Complete the first incomplete task
                    for i, task in enumerate(self.task_list):
                        if not task["completed"]:
                            self.complete_task(i)
                            break
                elif key == ord('r'):  # Generate daily report
                    self.generate_daily_report()

                # Save state periodically (e.g., every 5 minutes)
                if int(time.time()) % 300 == 0:
                    self.save_state()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            print("Saving final state and closing resources...")
            self.save_state()
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    def start(self):
        print("Starting Advanced Smart Home Assistant...")
        print("Controls:")
        print("  ESC: Quit the application")
        print("  F: Toggle focus mode")
        print("  T: Add a new task")
        print("  C: Complete the first incomplete task")
        print("  R: Generate daily report")
        self.run()

    def calculate_eye_strain_index(self):
        blink_rate = sum(self.blink_history) / len(self.blink_history)
        screen_time = sum(1 for activity in self.activity_history if activity == "WORKING") / len(
            self.activity_history)
        return (1 - blink_rate) * screen_time * 100  # Higher index means more eye strain

    def suggest_break(self):
        eye_strain_index = self.calculate_eye_strain_index()
        if eye_strain_index > 70:  # Threshold for suggesting a break
            return True
        return False

    def adjust_room_temperature(self, target_temp):
        diff = target_temp - self.room_temperature
        self.room_temperature += diff * 0.1  # Gradual adjustment

    def adjust_room_humidity(self, target_humidity):
        diff = target_humidity - self.room_humidity
        self.room_humidity += diff * 0.1  # Gradual adjustment

    def recommend_water_intake(self):
        if self.daily_water_intake < 2000:  # Assuming 2 liters as daily goal
            return 2000 - self.daily_water_intake
        return 0

    def analyze_productivity(self):
        work_time = sum(1 for activity in self.activity_history if activity == "WORKING")
        total_time = len(self.activity_history)
        productivity = work_time / total_time if total_time > 0 else 0
        return productivity * 100  # Return as percentage

    def generate_health_tips(self):
        tips = []
        if self.daily_step_count < 5000:
            tips.append("Try to increase your daily step count to at least 5000 steps.")
        if self.daily_water_intake < 2000:
            tips.append("Remember to stay hydrated. Aim for at least 2 liters of water per day.")
        if sum(self.posture_history) / len(self.posture_history) < 0.7:
            tips.append("Your posture needs improvement. Take breaks to stretch and align your spine.")
        if self.calculate_eye_strain_index() > 60:
            tips.append(
                "Your eyes may be strained. Remember the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds.")
        return tips

    def mood_based_recommendations(self):
        recent_emotions = list(self.emotion_history)[-20:]  # Last 20 emotion readings
        if recent_emotions.count("SAD") > 10:
            return "Your mood seems low. Consider taking a short walk or listening to uplifting music."
        elif recent_emotions.count("ANGRY") > 10:
            return "You seem stressed. Try some deep breathing exercises or take a short break."
        elif recent_emotions.count("HAPPY") > 15:
            return "You're in a great mood! This might be a good time for creative tasks or social interactions."
        return "Your mood seems balanced. Keep up the good work!"

if __name__ == "__main__":
    assistant = AdvancedSmartHomeAssistant()
    assistant.start()

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import pygetwindow as gw

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
session_duration = 60  # Define session duration, 60 seconds in this case

# Eye landmarks for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds and parameters
EAR_THRESHOLD = 0.2  # Blink detection threshold
CONSECUTIVE_FRAMES = 3  # Blink consecutive frame count
DROWSY_TIME_THRESHOLD = 5.0  # Seconds before drowsiness alert
VIDEO_DURATION = 60  # Total session duration in seconds

# Window switching tracking variables
last_window = None
window_history = []
started_tracking = False
restricted_apps = [
    "Google Chrome", "Brave", "Firefox",  # Browsers
    "YouTube", "VLC Media Player", "Media Player",  # Video players
    "Valorant", "GTA5", "Steam",  # Games
    "Discord", "Skype", "Chatgpt"  # Communication apps
]
opened_restricted_apps = set()
window_times = {}
session_start_time = time.time()

# Initialize counters
blink_count = 0
eye_frame_counter = 0
left_count, right_count, center_count = 0, 0, 0
drowsy_time = 0
drowsy_start_time = None
drowsy_episodes = 0

def calculate_ear(eye_landmarks):
    vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def get_active_window():
    window = gw.getActiveWindow()
    return window.title if window else None

def track_window_switches():
    global last_window, window_history, started_tracking, opened_restricted_apps, window_times

    try:
        while True:
            current_window = get_active_window()

            if current_window != last_window:
                if not started_tracking and "Visual Studio Code" in current_window:
                    last_window = current_window
                    print("Started tracking after Visual Studio Code.")
                    continue

                if not started_tracking:
                    started_tracking = True
                    print(f"Started tracking after: {current_window}")

                if current_window is not None:
                    if last_window and window_history:
                        time_spent = time.time() - window_history[-1][1]
                        if last_window in window_times:
                            window_times[last_window] += time_spent
                        else:
                            window_times[last_window] = time_spent

                    window_history.append((current_window, time.time()))
                    print(f"Switched to window: {current_window}")

                    for app in restricted_apps:
                        if app.lower() in current_window.lower():
                            opened_restricted_apps.add(app)

                last_window = current_window

            session_elapsed_time = time.time() - session_start_time

            if session_elapsed_time >= session_duration:
                if last_window and window_history:
                    time_spent = time.time() - window_history[-1][1]
                    if last_window in window_times:
                        window_times[last_window] += time_spent
                    else:
                        window_times[last_window] = time_spent

                total_time = sum(window_times.values())
                print("\nTime Spent on Each Window:")
                for window, time_spent in window_times.items():
                    print(f"{window}: {time_spent:.2f} seconds")

                vs_code_opened = False
                for window in window_times:
                    if "Visual Studio Code" in window:
                        vs_code_opened = True
                        vs_code_time = window_times[window]
                        break

                if vs_code_opened:
                    away_time = total_time - vs_code_time
                    away_percentage = (away_time / total_time) * 100 if total_time > 0 else 0
                    print(f"\nPercentage of time away from Visual Studio Code: {away_percentage:.2f}%")
                else:
                    print("\nVisual Studio Code was not opened during the session.")

                break

            print(f"Switch count: {len(window_history)}", end='\r')
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nProgram terminated.")
        display_window_history()

def display_window_history():
    global window_history, opened_restricted_apps
    print("\nWindow Switch History:")
    if window_history:
        for idx, (window, _) in enumerate(window_history, 1):
            print(f"{idx}. {window}")
    else:
        print("No windows were switched.")

    if opened_restricted_apps:
        print("\nWarning: The following restricted apps were opened during the session:")
        for app in opened_restricted_apps:
            print(f"- {app}")
    else:
        print("\nNo restricted apps were opened during the session.")

# Start video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

# Import necessary for threading
import threading

# Create a thread for window tracking
window_thread = threading.Thread(target=track_window_switches)
window_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            left_eye = [(int(face_landmarks.landmark[idx].x * frame_width), int(face_landmarks.landmark[idx].y * frame_height)) for idx in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[idx].x * frame_width), int(face_landmarks.landmark[idx].y * frame_height)) for idx in RIGHT_EYE]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Blink detection logic
            if avg_ear < EAR_THRESHOLD:
                eye_frame_counter += 1
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
            else:
                if eye_frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                eye_frame_counter = 0
                drowsy_start_time = None

            # Drowsiness detection logic
            if drowsy_start_time is not None:
                elapsed_drowsy_time = time.time() - drowsy_start_time
                if elapsed_drowsy_time >= DROWSY_TIME_THRESHOLD:
                    drowsy_time += elapsed_drowsy_time
                    drowsy_episodes += 1
                    drowsy_start_time = time.time()

            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Display EAR, Blink Count
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Get gaze direction landmarks
            left_eye_landmark = face_landmarks.landmark[33]
            right_eye_landmark = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            # Determine gaze direction
            if nose_tip.x < left_eye_landmark.x:
                direction = "Looking Left"
                left_count += 1
            elif nose_tip.x > right_eye_landmark.x:
                direction = "Looking Right"
                right_count += 1
            else:
                direction = "Looking Center"
                center_count += 1

            # Display gaze direction
            cv2.putText(frame, direction, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display gaze direction counters
    cv2.putText(frame, f"Left: {left_count}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right: {right_count}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Center: {center_count}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display drowsy episodes count
    cv2.putText(frame, f"Drowsy Episodes: {drowsy_episodes}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if 1 minute has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= VIDEO_DURATION:
        # Classify concentration level based on blink count
        concentration_level = "Focused" if blink_count < 10 else "Normal" if blink_count <= 20 else "Distracted"

        # Calculate gaze percentages
        total_gaze_count = left_count + right_count + center_count
        left_percentage = (left_count / total_gaze_count) * 100 if total_gaze_count > 0 else 0
        right_percentage = (right_count / total_gaze_count) * 100 if total_gaze_count > 0 else 0
        center_percentage = (center_count / total_gaze_count) * 100 if total_gaze_count > 0 else 0

        # Calculate drowsy time percentage
        drowsy_percentage = (drowsy_time / elapsed_time) * 100

        # Display final results
        cv2.putText(frame, f"Concentration: {concentration_level}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsy Time: {drowsy_percentage:.2f}%", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "End of Session", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Blink, Gaze & Drowsiness Tracker", frame)
        cv2.waitKey(2000)

        # Print the results in the terminal
        print(f"Total Blinks: {blink_count}")
        print(f"Concentration Level: {concentration_level}")
        print(f"Drowsy Time: {drowsy_time:.2f} seconds")
        print(f"Drowsy Time Percentage: {drowsy_percentage:.2f}%")
        print(f"Drowsy Episodes: {drowsy_episodes}")
        print("Gaze Direction Summary:")
        print(f"Left: {left_percentage:.2f}%, Right: {right_percentage:.2f}%, Center: {center_percentage:.2f}%")

        # Wait for window tracking thread to finish
        window_thread.join()
        break

    # Display the frame
    cv2.imshow("Blink, Gaze & Drowsiness Tracker", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        window_thread.join()
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
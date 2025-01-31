# blink + gaze
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark indices for eyes (MediaPipe Face Mesh landmarks)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Eye Aspect Ratio (EAR) threshold and consecutive frame threshold for blink detection
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 3

# Initialize blink counter and frame counters
blink_count = 0
eye_frame_counter = 0

# Initialize gaze direction counters
left_count = 0
right_count = 0
center_count = 0

# Time duration for the video capture (in seconds)
video_duration = 60  # 1 minute

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Start the webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    results = face_mesh.process(frame_rgb)

    # Frame height and width
    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get 2D landmarks for the left and right eye
            left_eye = [
                (int(face_landmarks.landmark[idx].x * frame_width),
                 int(face_landmarks.landmark[idx].y * frame_height))
                for idx in LEFT_EYE
            ]
            right_eye = [
                (int(face_landmarks.landmark[idx].x * frame_width),
                 int(face_landmarks.landmark[idx].y * frame_height))
                for idx in RIGHT_EYE
            ]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Average EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Detect blinks
            if avg_ear < EAR_THRESHOLD:
                eye_frame_counter += 1
            else:
                if eye_frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                eye_frame_counter = 0

            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Display EAR and Blink Count on the frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
            cv2.putText(frame, direction, (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display gaze direction counters
    cv2.putText(frame, f"Left: {left_count}", (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right: {right_count}", (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Center: {center_count}", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if 1 minute has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= video_duration:
        # Classify concentration level based on blink count
        if blink_count < 10:
            concentration_level = "Focused"
        elif blink_count <= 20:
            concentration_level = "Normal"
        else:
            concentration_level = "Distracted"

        # Calculate gaze percentages
        total_gaze_count = left_count + right_count + center_count
        if total_gaze_count > 0:
            left_percentage = (left_count / total_gaze_count) * 100
            right_percentage = (right_count / total_gaze_count) * 100
            center_percentage = (center_count / total_gaze_count) * 100
        else:
            left_percentage = right_percentage = center_percentage = 0

        # Display classification results
        cv2.putText(frame, f"Concentration: {concentration_level}", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Left: {left_percentage:.2f}%", (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Right: {right_percentage:.2f}%", (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Center: {center_percentage:.2f}%", (30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "End of Session", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Blink & Gaze Tracker", frame)
        cv2.waitKey(2000)  # Display for 2 seconds before closing

        # Print the results in the terminal
        print(f"Total Blinks: {blink_count}")
        print(f"Concentration Level (wrt Blink Counter) :  {concentration_level}")
        print("Gaze Direction Summary (over 1 minute session):")
        print(f"Left: {left_percentage:.2f}%")
        print(f"Right: {right_percentage:.2f}%")
        print(f"Center: {center_percentage:.2f}%")

        break

    # Display the frame
    cv2.imshow("Blink & Gaze Tracker", frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define face mesh model
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR Thresholds
EAR_THRESHOLD = 0.22  # Eye closure threshold
DROWSY_TIME_THRESHOLD = 5.0  # Time (seconds) before alert

# Track drowsiness
start_time = None
drowsy_start_time = None
is_drowsy = False
drowsy_time = 0  # Total drowsy time in seconds

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (A + B) / (2.0 * C)


# Eye landmarks (MediaPipe Mesh indices)
LEFT_EYE = [362, 385, 387, 263, 373, 380]  
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  

# Nose landmarks (for head position check)
NOSE = [1]  # You can use any nose point, this is just an example

# Capture webcam
cap = cv2.VideoCapture(0)

# Start the timer for the session
session_start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get face landmarks
            landmarks = {i: (lm.x * frame.shape[1], lm.y * frame.shape[0]) for i, lm in enumerate(face_landmarks.landmark)}

            # Calculate EAR
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            # Check if eyes are closed
            if avg_ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()  # Start timer for eyes closed

                # Check if the eyes are closed for more than the threshold
                elapsed_time = time.time() - start_time

                # Normal drowsiness detection
                if elapsed_time >= DROWSY_TIME_THRESHOLD:
                    # Increment drowsy time in chunks of the threshold
                    drowsy_time += (elapsed_time // DROWSY_TIME_THRESHOLD) * DROWSY_TIME_THRESHOLD

                    # Show the counter for seconds
                    cv2.putText(frame, f"Eyes Closed: {int(elapsed_time)}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Reset the start time for the next drowsy interval
                    start_time = time.time()

            else:
                start_time = None  # Reset timer if eyes are open
                is_drowsy = False  # Reset drowsy state

            # Calculate total drowsiness count (in multiples of the threshold)
            drowsy_count = int(drowsy_time // DROWSY_TIME_THRESHOLD)

            # Show the drowsy count
            cv2.putText(frame, f"Drowsiness Episodes: {drowsy_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Calculate session time and end after 1 minute
    session_elapsed_time = time.time() - session_start_time
    if session_elapsed_time >= 60:
        # Calculate drowsy time percentage
        drowsy_percentage = (drowsy_time / session_elapsed_time) * 100

        # Display the session result on screen
        cv2.putText(frame, f"Drowsy Time: {drowsy_percentage:.2f}%", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Print the result in the terminal
        print(f"Drowsy Time: {drowsy_time:.2f} seconds")
        print(f"Total Session Time: {session_elapsed_time:.2f} seconds")
        print(f"Percentage of Drowsy Time: {drowsy_percentage:.2f}%")
        print(f"Drowsiness Episodes: {drowsy_count}")

        # Break the loop after 1 minute
        break

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

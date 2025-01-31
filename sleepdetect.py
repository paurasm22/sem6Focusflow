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
DROWSY_COUNT_THRESHOLD = 5.0  # Time (seconds) before counting a drowsiness episode

# Track drowsiness
start_time = None
is_drowsy = False
drowsy_count = 0  # Drowsiness counter

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
                    start_time = time.time()  # Start timer

                elapsed_time = time.time() - start_time

                # Check if head is tilted down (resting on headrest)
                nose_y = landmarks[NOSE[0]][1]
                left_eye_y = landmarks[LEFT_EYE[1]][1]
                right_eye_y = landmarks[RIGHT_EYE[1]][1]

                # If the nose is much lower than the eyes, we assume the head is tilted down
                if nose_y > (left_eye_y + right_eye_y) / 2.0 + 50:  # Adjust the value based on testing
                    continue  # Skip drowsiness check if head is tilted down

                # Normal drowsiness detection
                if elapsed_time > DROWSY_TIME_THRESHOLD and not is_drowsy:
                    cv2.putText(frame, "⚠️ DROWSINESS ALERT! ⚠️", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    is_drowsy = True  # Set drowsy state
                    
                    # Increment the drowsy count when threshold is exceeded
                    drowsy_count += 1
                
                # Show the counter for seconds
                cv2.putText(frame, f"Eyes Closed: {int(elapsed_time)}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                start_time = None  # Reset timer if eyes are open
                is_drowsy = False  # Reset drowsy state

            # Show the drowsy count
            cv2.putText(frame, f"Drowsiness Episodes: {drowsy_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Display frame
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

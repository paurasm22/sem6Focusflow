import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

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
left_eye_frame_counter = 0
right_eye_frame_counter = 0

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Start the webcam
cap = cv2.VideoCapture(0)

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
                left_eye_frame_counter += 1
            else:
                if left_eye_frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                left_eye_frame_counter = 0

            # Draw landmarks and EAR on the frame
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Display EAR and Blink Count on the frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Blink Counter", frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize Mediapipe face mesh detector
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define constants
NOD_THRESHOLD = 2  # Number of consecutive frames the nod must persist
nod_count = 0  # Counter to track the number of nods
consecutive_nods = 0  # To count consecutive nods
prev_forehead_y = None  # Store previous forehead y-coordinate for comparison
prev_nose_y = None  # Store previous nose y-coordinate for comparison

# Start video capture
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for better mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get facial landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw face mesh landmarks
                mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Get the position of the forehead (landmark 10) and nose (landmark 2)
                forehead = landmarks.landmark[10]
                nose = landmarks.landmark[2]

                # Track the vertical movement of the forehead and nose
                current_forehead_y = forehead.y
                current_nose_y = nose.y

                # Compare with previous forehead and nose y-coordinates to detect significant movement
                if prev_forehead_y is not None and prev_nose_y is not None:
                    # Detect significant movement in both forehead and nose positions
                    if abs(prev_forehead_y - current_forehead_y) > 0.003 or abs(prev_nose_y - current_nose_y) > 0.003:
                        consecutive_nods += 1
                    else:
                        consecutive_nods = 0  # Reset if no significant movement detected

                # If we have enough consecutive nods, count it as a valid nod
                if consecutive_nods >= NOD_THRESHOLD:
                    nod_count += 1
                    consecutive_nods = 0  # Reset after counting a nod

                # Update previous forehead and nose y-coordinates for next comparison
                prev_forehead_y = current_forehead_y
                prev_nose_y = current_nose_y

                # Show nod count on the frame
                cv2.putText(frame, f"Nod Count: {nod_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the resulting frame
        cv2.imshow("Nod Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Capture video stream from webcam
cap = cv2.VideoCapture(0)

# Initialize counters for gaze direction
left_count = 0
right_count = 0
center_count = 0

# Start the timer
start_time = time.time()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror-like effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the result
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of key facial landmarks
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            # Calculate head position
            if nose_tip.x < left_eye.x:
                direction = "Looking Left"
                left_count += 1
            elif nose_tip.x > right_eye.x:
                direction = "Looking Right"
                right_count += 1
            else:
                direction = "Looking Center"
                center_count += 1

            # Put the direction text on the frame
            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw the face landmarks on the frame
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the gaze direction counters on the frame
    cv2.putText(frame, f"Left: {left_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Right: {right_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Center: {center_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check if 1 minute (60 seconds) has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        # End the session and calculate the percentages
        total_gaze_count = left_count + right_count + center_count
        if total_gaze_count > 0:
            left_percentage = (left_count / total_gaze_count) * 100
            right_percentage = (right_count / total_gaze_count) * 100
            center_percentage = (center_count / total_gaze_count) * 100
        else:
            left_percentage = right_percentage = center_percentage = 0

        # Display the final gaze direction percentages
        cv2.putText(frame, f"Left: {left_percentage:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right: {right_percentage:.2f}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Center: {center_percentage:.2f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the result for 2 seconds
        cv2.imshow('Frame', frame)
        cv2.waitKey(2000)  # Wait for 2 seconds before closing

        # Print the results in the terminal
        print("Gaze Direction Summary (over 1 minute session):")
        print(f"Left: {left_percentage:.2f}%")
        print(f"Right: {right_percentage:.2f}%")
        print(f"Center: {center_percentage:.2f}%")
        
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()

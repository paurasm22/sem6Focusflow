import cv2

# Initialize webcam (use '0' for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam is successfully opened.")

# Release the webcam after use
cap.release()

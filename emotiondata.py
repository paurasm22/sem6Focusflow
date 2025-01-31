import cv2
from deepface import DeepFace
import time

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam (use '0' for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize emotion counters
emotion_count = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0,
}

# Start session timer
session_start_time = time.time()
session_duration = 60  # 60 seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.resize(frame, (320, 240))
    # Convert the frame from BGR (OpenCV format) to RGB (DeepFace format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Detect faces in the frame using Haar cascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Crop the face region for emotion analysis
            face_region = rgb_frame[y:y + h, x:x + w]

            # Analyze the face for emotion detection using DeepFace
            result = DeepFace.analyze(rgb_frame, actions=["emotion"], detector_backend="mtcnn")

            # Extract the dominant emotion
            emotion = result[0]["dominant_emotion"]

            # Increment the emotion counter
            if emotion in emotion_count:
                emotion_count[emotion] += 1

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the emotion on the frame
            cv2.putText(
                frame,
                f"{emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    except Exception as e:
        # Handle cases where DeepFace cannot process a face region
        print(f"Error in DeepFace analysis: {e}")

    # Calculate session elapsed time
    session_elapsed_time = time.time() - session_start_time

    # If the session time exceeds 1 minute, break out of the loop
    if session_elapsed_time >= session_duration:
        # Calculate emotion percentage
        total_emotions = sum(emotion_count.values())
        emotion_percentages = {emotion: (count / total_emotions) * 100 if total_emotions > 0 else 0 
                               for emotion, count in emotion_count.items()}

        # Display emotion percentages at the end of the session
        print("\nEmotion Statistics for the Session:")
        for emotion, percentage in emotion_percentages.items():
            print(f"{emotion.capitalize()}: {percentage:.2f}%")

        # Break the loop after 1 minute
        break

    # Display the captured frame with rectangles and emotion text
    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

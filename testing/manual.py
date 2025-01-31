import cv2
from deepface import DeepFace

# Specify the image file name
image_path = "testing\\fear.jpg"  # Replace with your actual image file name

# Load the image using OpenCV
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Resize the image to a smaller size (you can adjust the size based on your requirements)
image = cv2.resize(image, (400, 400))  # Resize to 400x400, you can adjust this size

# Convert the image from BGR (OpenCV format) to RGB (DeepFace format)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform emotion analysis using DeepFace
try:
    result = DeepFace.analyze(rgb_image, actions=["emotion"], detector_backend="mtcnn")

    # Extract the dominant emotion
    emotion = result[0]["dominant_emotion"]

    # Display the emotion on the image
    cv2.putText(
        image,
        f"Emotion: {emotion}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the image with emotion text
    cv2.imshow("Emotion Detection", image)

    # Wait for a key press to close the image window
    cv2.waitKey(0)

    # Close the OpenCV window
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error in DeepFace analysis: {e}")

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

#load the model
model = load_model(r'C:\Users\prajw\PycharmProjects\python_athul\ann project/facial_expression_model.h5')
print(model.summary())

emotions = ['Happiness', 'Sadness', 'Anger', 'Fear','disgust','Surprise', 'Contempt']
predicted_emotion=""


def capture_face_photo():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the default camera (index 0)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the region of interest (ROI) for the face
            face_roi = frame[y:y+h, x:x+w]

            # Release the video capture object
            cap.release()

            # Close all OpenCV windows
            cv2.destroyAllWindows()

            return face_roi

        # Display the resulting frame with face detection
        cv2.imshow('Face Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




# Function to preprocess the image
def preprocess_frame(frame):
    input_shape = (48, 48)

    # Resize the frame to the model's input shape
    resized_frame = cv2.resize(frame, input_shape, interpolation=cv2.INTER_AREA)

    # Convert the frame to a numpy array
    frame_array = np.array(resized_frame)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

    # Convert to grayscale if the model expects a single-channel input
    if frame_array.shape[-1] == 3:
        frame_array = np.dot(frame_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Expand dimensions to match the model's input shape (add a batch dimension)
    frame_array = np.expand_dims(frame_array, axis=0)

    # Normalize pixel values to be between 0 and 1
    frame_array = frame_array.astype('float32') / 255.0

    # Reshape to match the model's input shape (add channel dimension)
    frame_array = np.reshape(frame_array, (1, input_shape[0], input_shape[1], 1))

    return frame_array


# Call the function to capture a face photo and get the image of the face
captured_face = capture_face_photo()

# You can use the 'captured_face' image as needed (e.g., save it to a file, process it further, etc.)
cv2.imshow('Captured Face', captured_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
procesesd_frame = preprocess_frame(captured_face)
plt.figure(figsize=(3, 3))
plt.imshow(procesesd_frame.reshape(48, 48), cmap='gray')  # Assuming grayscale images
emotion_label = "unknown"
plt.title("processed image")
plt.show()
# Make predictions using the model
predictions = model.predict(procesesd_frame)
print(predictions)
index = np.argmax(predictions)
# Get the predicted emotion
predicted_emotion = emotions[index]
print(predicted_emotion)





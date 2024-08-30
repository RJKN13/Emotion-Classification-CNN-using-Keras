import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the face detector and the trained emotion detection model
face_classifier = cv2.CascadeClassifier(r'C:\Users\rajku\OneDrive\Dokument\GitHub\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\rajku\OneDrive\Dokument\GitHub\Emotion_Detection_CNN\my_model.keras')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)
image_count = 0  # Counter for saved images

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If the frame is not read properly, continue to the next iteration
    if not ret:
        continue
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Extract the region of interest (ROI) of the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize the ROI and prepare it for the model
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make a prediction using the model
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        
        # Display the label on the frame
        label_position = (x, y - 10)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    # If any key is pressed, save the current frame
    if key== ord('s'):
        image_count += 1
        image_filename = f"detected_face_{image_count}.jpg"
        cv2.imwrite(image_filename, frame)
    
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

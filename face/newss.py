from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np

# Load models
face_classifier = cv2.CascadeClassifier('/workspaces/project/face/haarcascade_frontalface_default.xml')
emotion_model = load_model('/workspaces/project/face/emotion_detection_model_50epochs.h5')
#age_model = load_model('/workspaces/project/face/age_model_3epochs.h5')
gender_model = load_model('/workspaces/project/face/gender_model_3epochs.h5')

# Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Video file path
video_path = '/workspaces/project/face/bid.mp4'  
output_path = '/workspaces/project/face/output.mp4'  # Path to save the output video
cap = cv2.VideoCapture(video_path)

# Get the video width, height, and frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the video file.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Emotion Prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_model.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gender Prediction
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
        gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
        gender_label = gender_labels[gender_predict[0]]
        gender_label_position = (x, y + h + 50)
        cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Age Prediction
        """
        age_predict = age_model.predict(np.array(roi_color).reshape(-1, 200, 200, 3))
        age = round(age_predict[0, 0])
        age_label_position = (x + h, y + h)
        cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        """

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f"Video saved to {output_path}")

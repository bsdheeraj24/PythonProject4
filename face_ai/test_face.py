import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("face_model.keras")

# Class labels (MUST match training)
labels = {
    0: "Anil",
    1: "Dheeraj"
}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Cascade loaded:", not face_cascade.empty())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)
        class_id = np.argmax(preds)
        confidence = preds[0][class_id]

        if confidence < 0.75:
            label = "Unknown"
        else:
            label = f"{labels[class_id]} ({confidence:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

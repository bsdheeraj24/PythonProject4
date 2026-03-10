import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X = []
y = []
labels = {}
label_id = 0

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    labels[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (96, 96))
        img = img / 255.0

        X.append(img)
        y.append(label_id)

    label_id += 1

X = np.array(X)
y = np.array(y)

print("Dataset loaded:", X.shape)
print("Classes:", labels)

model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20, batch_size=8)

model.save("face_model.keras")

print("✅ Model training completed and saved")

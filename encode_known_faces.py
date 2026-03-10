# encode_known_faces.py
import os
import face_recognition
import pickle

KNOWN_DIR = "known_faces"
OUTPUT = "known_faces_encodings.pkl"

encodings = []
names = []

for person in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print(f"No face found in {img_path}, skipping.")
            continue
        # take first face found
        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        encodings.append(encoding)
        names.append(person)
        print(f"Encoded {img_path} -> {person}")

# save to disk
import pickle
with open(OUTPUT, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)
print("Saved encodings to", OUTPUT)

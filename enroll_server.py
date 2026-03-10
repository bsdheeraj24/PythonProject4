import base64, io, os, pickle
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image

ENC_FILE = "encodings.pkl"
KNOWN_DIR = "known_faces"
SAMPLES = 10

os.makedirs(KNOWN_DIR, exist_ok=True)
app = Flask(__name__)

count = 0
person_name = None
known_encodings = []
known_names = []

@app.route("/start_enroll", methods=["POST"])
def start_enroll():
    global count, person_name
    data = request.json
    person_name = data["name"]
    count = 0
    os.makedirs(f"{KNOWN_DIR}/{person_name}", exist_ok=True)
    return jsonify({"status": "started"})

@app.route("/capture", methods=["POST"])
def capture():
    global count

    img = base64.b64decode(request.json["image"])
    frame = np.array(Image.open(io.BytesIO(img)).convert("RGB"))

    encs = face_recognition.face_encodings(frame)
    if not encs:
        return jsonify({"status": "no_face"})

    known_encodings.append(encs[0])
    known_names.append(person_name)

    count += 1
    Image.fromarray(frame).save(f"{KNOWN_DIR}/{person_name}/{count}.jpg")

    if count >= SAMPLES:
        with open(ENC_FILE, "wb") as f:
            pickle.dump({"enc": known_encodings, "names": known_names}, f)
        return jsonify({"status": "done"})

    return jsonify({"status": "saved", "count": count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

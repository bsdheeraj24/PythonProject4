import base64, io, pickle, csv, os
from datetime import datetime
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image

ENC_FILE = "encodings.pkl"
ATT_DIR = "attendance"
os.makedirs(ATT_DIR, exist_ok=True)

with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)
known_enc = data["enc"]
known_names = data["names"]

LAST = {"status": "waiting"}
app = Flask(__name__)

@app.route("/last")
def last():
    return jsonify(LAST)

@app.route("/capture", methods=["POST"])
def capture():
    global LAST

    img = base64.b64decode(request.json["image"])
    frame = np.array(Image.open(io.BytesIO(img)).convert("RGB"))

    encs = face_recognition.face_encodings(frame)
    if not encs:
        LAST = {"status": "no_face"}
        return jsonify(LAST)

    enc = encs[0]
    matches = face_recognition.compare_faces(known_enc, enc, 0.5)

    if True not in matches:
        LAST = {"status": "unknown"}
        return jsonify(LAST)

    idx = matches.index(True)
    name = known_names[idx]
    conf = int((1 - face_recognition.face_distance(known_enc, enc)[idx]) * 100)

    now = datetime.now()
    file = f"{ATT_DIR}/{name}.csv"
    status = "IN"

    if os.path.exists(file):
        with open(file) as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                status = "OUT" if rows[-1][3] == "IN" else "IN"

    with open(file, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["Date","Time","Name","Status"])
        w.writerow([now.date(), now.time(), name, status])

    LAST = {"status":"recognized","name":name,"confidence":conf}
    return jsonify(LAST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

import base64
import io
import os
import csv
import pickle
import time
from datetime import datetime

from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image

# ================= CONFIG =================
ENC_FILE = "known_faces_encodings.pkl"
KNOWN_DIR = "known_faces"
ATTEND_DIR = "attendance"

ENROLL_SAMPLES = 10          # Number of images per person
ENROLL_COOLDOWN = 20         # Seconds after enrollment
MIN_GAP_SECONDS = 60         # Gap between IN <-> OUT
# =========================================

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(ATTEND_DIR, exist_ok=True)

app = Flask(__name__)

# ---------- GLOBAL STATE ----------
MODE = {"type": "attend", "name": None}
ENROLL_COUNT = 0
ENROLL_TIME = None

# ---------- LOAD ENCODINGS ----------
if os.path.exists(ENC_FILE):
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
else:
    known_encodings = []
    known_names = []

# ---------- LAST ATTENDANCE ----------
def get_last_entry(person):
    file = f"{ATTEND_DIR}/{person}.csv"
    if not os.path.exists(file):
        return None, None

    with open(file, "r") as f:
        rows = list(csv.reader(f))
        if len(rows) <= 1:
            return None, None

        last = rows[-1]
        last_time = datetime.strptime(
            f"{last[0]} {last[1]}", "%Y-%m-%d %H:%M:%S"
        )
        return last[3], last_time


# ---------- SET MODE ----------
@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE, ENROLL_COUNT, ENROLL_TIME
    MODE = request.json
    ENROLL_COUNT = 0
    ENROLL_TIME = None
    print("[MODE SET]", MODE)
    return jsonify(MODE)


# ---------- CAPTURE ----------
@app.route("/capture", methods=["GET", "POST"])
def capture():
    global MODE, ENROLL_COUNT, ENROLL_TIME
    global known_encodings, known_names

    # ESP32 polling current mode
    if request.method == "GET":
        return jsonify(MODE)

    # Decode image
    img_bytes = base64.b64decode(request.json["image"])
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(img_pil)

    # Face detection
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) == 0:
        return jsonify({"status": "NO_FACE"})

    if len(face_encodings) > 1:
        return jsonify({"status": "MULTIPLE_FACES"})

    face = face_encodings[0]

    # ================= ENROLL MODE =================
    if MODE["type"] == "enroll":
        person = MODE["name"]
        person_dir = f"{KNOWN_DIR}/{person}"
        os.makedirs(person_dir, exist_ok=True)

        img_path = f"{person_dir}/img_{ENROLL_COUNT + 1}.jpg"
        img_pil.save(img_path)

        known_encodings.append(face)
        known_names.append(person)
        ENROLL_COUNT += 1

        print(f"[ENROLL] {person} {ENROLL_COUNT}/{ENROLL_SAMPLES}")

        if ENROLL_COUNT >= ENROLL_SAMPLES:
            with open(ENC_FILE, "wb") as f:
                pickle.dump(
                    {"encodings": known_encodings, "names": known_names}, f
                )

            MODE = {"type": "attend", "name": None}
            ENROLL_TIME = time.time()

            print(f"[ENROLL COMPLETE] {person}")

            return jsonify({
                "status": "ENROLL_COMPLETE",
                "cooldown": ENROLL_COOLDOWN
            })

        return jsonify({
            "status": "ENROLLING",
            "sample": ENROLL_COUNT
        })

    # ================= ATTEND MODE =================
    # Enrollment cooldown
    if ENROLL_TIME and (time.time() - ENROLL_TIME) < ENROLL_COOLDOWN:
        remaining = int(ENROLL_COOLDOWN - (time.time() - ENROLL_TIME))
        return jsonify({
            "status": "WAIT",
            "seconds": remaining
        })

    matches = face_recognition.compare_faces(
        known_encodings, face, tolerance=0.5
    )

    if True not in matches:
        return jsonify({"status": "UNKNOWN"})

    idx = matches.index(True)
    person = known_names[idx]

    last_status, last_time = get_last_entry(person)
    now = datetime.now()

    if last_status is None:
        status = "IN"
    else:
        diff = (now - last_time).total_seconds()
        if diff < MIN_GAP_SECONDS:
            return jsonify({
                "status": "WAIT",
                "seconds": int(MIN_GAP_SECONDS - diff)
            })
        status = "OUT" if last_status == "IN" else "IN"

    # Save attendance
    file = f"{ATTEND_DIR}/{person}.csv"
    new_file = not os.path.exists(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Date", "Time", "Name", "Status"])
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            person,
            status
        ])

    print(f"[ATTENDANCE] {person} -> {status}")

    return jsonify({
        "status": "ATTENDANCE_MARKED",
        "name": person,
        "entry": status
    })


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

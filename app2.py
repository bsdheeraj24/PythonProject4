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

ENROLL_SAMPLES = 10
ENROLL_COOLDOWN = 20
MIN_GAP_SECONDS = 10
# =========================================

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(ATTEND_DIR, exist_ok=True)

app = Flask(__name__)

# ================= GLOBAL STATE =================
MODE = {"type": "attend", "name": None}
ENROLL_COUNT = 0
ENROLL_TIME = None

LAST_RESULT = {
    "status": "IDLE",
    "name": "",
    "confidence": 0
}

# ================= LOAD ENCODINGS =================
if os.path.exists(ENC_FILE):
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
else:
    known_encodings = []
    known_names = []

print(f"[SERVER] Loaded {len(known_names)} known faces")

print("\n===================================")
print(" Smart Attendance Flask Server")
print("===================================")


# ================= HELPER =================
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


# ================= MODE SET =================
@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE, ENROLL_COUNT, ENROLL_TIME

    data = request.get_json(force=True, silent=True)

    print("\n==============================")
    print("[MODE RECEIVED FROM ESP32]")
    print(data)
    print("==============================")

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # Accept both "type" or "mode"
    if "type" in data:
        MODE["type"] = data.get("type", "attend")
        MODE["name"] = data.get("name", None)

    elif "mode" in data:
        MODE["type"] = data.get("mode", "attend")
        MODE["name"] = data.get("name", None)

    else:
        return jsonify({"error": "Invalid mode format"}), 400

    ENROLL_COUNT = 0
    ENROLL_TIME = None

    print(f"[SERVER MODE SET] {MODE}")

    return jsonify(MODE), 200


# ================= STATUS =================
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "mode": MODE.get("type", "unknown"),
        "name": MODE.get("name", ""),
        "enroll_count": ENROLL_COUNT,
        "last_result": LAST_RESULT
    })


# ================= DUMMY ENDPOINTS (SAFE) =================
@app.route("/should_capture")
def should_capture():
    return jsonify({"capture": True})


@app.route("/last_recognition")
def last_recognition():
    return jsonify(LAST_RESULT)


# ================= CAPTURE =================
@app.route("/capture", methods=["POST"])
def capture():
    global ENROLL_COUNT, ENROLL_TIME, LAST_RESULT
    global known_encodings, known_names

    if "image" not in request.json:
        return jsonify({"status": "NO_IMAGE"})

    img_bytes = base64.b64decode(request.json["image"])
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(img_pil)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) == 0:
        LAST_RESULT = {"status": "NO_FACE", "name": "", "confidence": 0}
        return jsonify(LAST_RESULT)

    if len(face_encodings) > 1:
        LAST_RESULT = {"status": "MULTIPLE_FACES", "name": "", "confidence": 0}
        return jsonify(LAST_RESULT)

    face = face_encodings[0]

    # ================= ENROLL =================
    if MODE["type"] == "enroll":
        person = MODE["name"]

        if not person:
            return jsonify({"status": "NO_NAME"})

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

            MODE["type"] = "attend"
            MODE["name"] = None
            ENROLL_TIME = time.time()

            LAST_RESULT = {
                "status": "ENROLL_COMPLETE",
                "name": person,
                "confidence": 100
            }

            print(f"[ENROLL COMPLETE] {person}")

            return jsonify(LAST_RESULT)

        return jsonify({
            "status": "ENROLLING",
            "sample": ENROLL_COUNT
        })

    # ================= ATTEND =================
    if ENROLL_TIME and (time.time() - ENROLL_TIME) < ENROLL_COOLDOWN:
        remaining = int(ENROLL_COOLDOWN - (time.time() - ENROLL_TIME))
        return jsonify({"status": "WAIT", "seconds": remaining})

    matches = face_recognition.compare_faces(
        known_encodings, face, tolerance=0.5
    )

    distances = face_recognition.face_distance(known_encodings, face)

    if True not in matches:
        LAST_RESULT = {"status": "UNKNOWN", "name": "", "confidence": 0}
        return jsonify(LAST_RESULT)

    idx = matches.index(True)
    person = known_names[idx]
    confidence = int((1 - distances[idx]) * 100)

    last_status, last_time = get_last_entry(person)
    now = datetime.now()

    if last_status is None:
        status = "IN"
    else:
        diff = (now - last_time).total_seconds()
        if diff < MIN_GAP_SECONDS:
            return jsonify({"status": "WAIT", "seconds": int(MIN_GAP_SECONDS - diff)})
        status = "OUT" if last_status == "IN" else "IN"

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

    LAST_RESULT = {
        "status": "ATTENDANCE_MARKED",
        "name": person,
        "entry": status,
        "confidence": confidence
    }

    print(f"[ATTENDANCE] {person} -> {status} ({confidence}%)")

    return jsonify(LAST_RESULT)


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

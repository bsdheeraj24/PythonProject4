import os
import csv
import base64
import io
import pickle
import shutil
import json
import binascii
from collections import OrderedDict
from datetime import datetime
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    redirect, session, send_file
)

import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.security import generate_password_hash, check_password_hash

import face_recognition
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= CONFIG =================
ENC_FILE = "known_faces_encodings.pkl"
KNOWN_DIR = "known_faces"
ENROLL_SAMPLES = 10
MIN_GAP_SECONDS = 20

ESP32_CAM_IP = None

# ================= FOLDERS =================
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ================= FIREBASE =================
def _load_firebase_credentials_from_env():
    """Load Firebase credentials from JSON or base64(JSON) env var."""
    raw = os.environ.get("FIREBASE_CREDENTIALS", "").strip()
    if not raw:
        raise RuntimeError("FIREBASE_CREDENTIALS is not set")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            data = json.loads(decoded)
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS must be valid JSON or base64-encoded JSON"
            ) from exc

    if isinstance(data, dict) and "private_key" in data and isinstance(data["private_key"], str):
        data["private_key"] = data["private_key"].replace("\\n", "\n")

    return data


FIREBASE_INIT_ERROR = None
db = None

try:
    firebase_creds = _load_firebase_credentials_from_env()
    cred = credentials.Certificate(firebase_creds)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as exc:
    FIREBASE_INIT_ERROR = str(exc)
    print(f"Firebase init failed: {FIREBASE_INIT_ERROR}")

# ================= FLASK =================
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "attendance_secret_key_fallback")


@app.before_request
def ensure_backend_ready():
    # Keep Render health checks from failing hard if Firebase env is misconfigured.
    if db is None:
        return jsonify({
            "status": "CONFIG_ERROR",
            "message": "Firebase is not configured correctly.",
            "details": FIREBASE_INIT_ERROR,
        }), 503

# ================= GLOBAL STATE =================
MODE = {"type": "idle", "name": None}
ENROLL_COUNT = 0

LAST_RESULT = {
    "status": "IDLE",
    "name": "",
    "entry": "",
    "confidence": 0
}

# ================= HELPERS =================
def _remove_person_from_meta(name):
    meta_ref = db.collection("metadata").document("attendance_persons")
    meta_doc = meta_ref.get()
    if meta_doc.exists:
        names = meta_doc.to_dict().get("names", [])
        if name in names:
            names.remove(name)
            meta_ref.update({"names": names})

# ================= USERS =================
def load_users():
    users = {}
    docs = db.collection("users").stream()
    for doc in docs:
        users[doc.id] = doc.to_dict()
    return users

# ================= LOAD ENCODINGS =================
if os.path.exists(ENC_FILE):
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
else:
    known_encodings = []
    known_names = []

# ================= LOGIN REQUIRED =================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrap

# ================= FILTER =================
@app.template_filter("ddmmyyyy")
def ddmmyyyy(date_str):
    try:
        y, m, d = date_str.split("-")
        return f"{d}-{m}-{y}"
    except:
        return date_str

# ================= ESP32-CAM REGISTER =================
@app.route("/esp32_register", methods=["POST"])
def esp32_register():
    global ESP32_CAM_IP
    data = request.get_json(force=True, silent=True) or {}
    ESP32_CAM_IP = data.get("ip")
    print("ESP32-CAM Registered IP:", ESP32_CAM_IP)
    return jsonify({"status": "ok", "ip": ESP32_CAM_IP})

# ================= AUTH =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        doc = db.collection("users").document(username).get()
        if doc.exists:
            user = doc.to_dict()
            if check_password_hash(user["password_hash"], password):
                session["user"] = username
                session["role"] = user["role"]
                return redirect("/dashboard")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= DASHBOARD =================
@app.route("/dashboard")
@login_required
def dashboard():
    stream_url = f"http://{ESP32_CAM_IP}:81/stream" if ESP32_CAM_IP else ""
    return render_template("dashboard.html", stream_url=stream_url)

# ================= USERS =================
@app.route("/users", methods=["GET", "POST"])
@login_required
def manage_users():
    if session.get("role") != "admin":
        return "Access denied"

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username and password:
            db.collection("users").document(username).set({
                "password_hash": generate_password_hash(password),
                "role": "user",
            })

    users = load_users()
    return render_template("users.html", users=users)

@app.route("/delete_user/<username>")
@login_required
def delete_user(username):
    if session.get("role") != "admin":
        return "Access denied"

    if username == session.get("user"):
        return "Cannot delete current admin"

    db.collection("users").document(username).delete()
    return redirect("/users")

# ================= MODE (from ESP32 Nextion) =================
@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE, ENROLL_COUNT, LAST_RESULT

    data = request.get_json(force=True, silent=True) or {}
    print("✅ /mode received:", data)

    new_mode = data.get("mode") or data.get("type")

    if new_mode in ["idle", "enroll", "attend"]:
        MODE["type"] = new_mode

    MODE["name"] = data.get("name", None)
    ENROLL_COUNT = 0

    LAST_RESULT = {
        "status": MODE["type"].upper(),
        "name": MODE["name"] or "",
        "entry": "",
        "confidence": 0
    }

    return jsonify({"status": "ok", "mode": MODE})

@app.route("/status")
def status():
    return jsonify({
        "mode": MODE["type"],
        "name": MODE["name"],
        "enroll_count": ENROLL_COUNT,
        "last": LAST_RESULT
    })

@app.route("/last_recognition")
def last_recognition():
    return jsonify(LAST_RESULT)

# ================= CAPTURE (from ESP32-CAM) =================
@app.route("/capture", methods=["POST"])
def capture():
    global ENROLL_COUNT, LAST_RESULT
    global known_encodings, known_names

    data = request.get_json(force=True, silent=True) or {}
    if "image" not in data:
        return jsonify({"status": "NO_IMAGE"})

    try:
        img_bytes = base64.b64decode(data["image"])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return jsonify({"status": "BAD_IMAGE"})

    frame = np.array(img_pil)

    encodings = face_recognition.face_encodings(frame)
    if not encodings:
        LAST_RESULT = {"status": "NO_FACE", "name": "", "entry": "", "confidence": 0}
        return jsonify(LAST_RESULT)

    face = encodings[0]

    # -------- ENROLL --------
    if MODE["type"] == "enroll":
        name = MODE["name"]
        if not name:
            return jsonify({"status": "NO_NAME"})

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # ✅ Save JPG samples
        img_path = os.path.join(person_dir, f"img_{ENROLL_COUNT + 1}.jpg")
        img_pil.save(img_path, "JPEG")

        known_encodings.append(face)
        known_names.append(name)
        ENROLL_COUNT += 1

        if ENROLL_COUNT >= ENROLL_SAMPLES:
            with open(ENC_FILE, "wb") as f:
                pickle.dump({"encodings": known_encodings, "names": known_names}, f)

            MODE["type"] = "idle"
            MODE["name"] = None

            LAST_RESULT = {"status": "ENROLL_COMPLETE", "name": name, "entry": "", "confidence": 100}
            return jsonify(LAST_RESULT)

        LAST_RESULT = {"status": "ENROLLING", "name": name, "entry": "", "confidence": 0}
        return jsonify({"status": "ENROLLING", "count": ENROLL_COUNT})

    # -------- ATTEND --------
    matches = face_recognition.compare_faces(known_encodings, face, tolerance=0.5)
    if True not in matches:
        LAST_RESULT = {"status": "UNKNOWN", "name": "", "entry": "", "confidence": 0}
        return jsonify(LAST_RESULT)

    idx = matches.index(True)
    name = known_names[idx]
    confidence = 85

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    entry = "IN"

    # Query today's entries for this person
    today_docs = (
        db.collection("attendance")
        .where("name", "==", name)
        .where("date", "==", today)
        .order_by("time")
        .get()
    )

    if today_docs:
        last_doc = today_docs[-1].to_dict()
        last_status = last_doc["status"]
        last_time = datetime.strptime(
            f"{last_doc['date']} {last_doc['time']}", "%Y-%m-%d %H:%M:%S"
        )
        diff = (now - last_time).total_seconds()

        if diff < MIN_GAP_SECONDS:
            LAST_RESULT = {"status": "WAIT", "name": name, "entry": "", "confidence": 0}
            return jsonify({"status": "WAIT", "seconds": int(MIN_GAP_SECONDS - diff)})

        entry = "OUT" if last_status == "IN" else "IN"

    # Write new attendance record
    db.collection("attendance").add({
        "name": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": entry,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    # Update metadata (add person if not already tracked)
    meta_ref = db.collection("metadata").document("attendance_persons")
    meta_doc = meta_ref.get()
    if meta_doc.exists:
        names_list = meta_doc.to_dict().get("names", [])
        if name not in names_list:
            names_list.append(name)
            meta_ref.update({"names": names_list})
    else:
        meta_ref.set({"names": [name]})

    LAST_RESULT = {"status": "ATTENDANCE_MARKED", "name": name, "entry": entry, "confidence": confidence}
    return jsonify(LAST_RESULT)

# ================= FACES =================
@app.route("/faces")
@login_required
def faces():
    persons = [
        d for d in os.listdir(KNOWN_DIR)
        if os.path.isdir(os.path.join(KNOWN_DIR, d))
    ]
    return render_template("faces.html", users=persons)

@app.route("/delete/<name>")
@login_required
def delete_face(name):
    global known_encodings, known_names

    idxs = [i for i, n in enumerate(known_names) if n == name]
    for i in sorted(idxs, reverse=True):
        known_encodings.pop(i)
        known_names.pop(i)

    with open(ENC_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    shutil.rmtree(os.path.join(KNOWN_DIR, name), ignore_errors=True)

    # Delete attendance records from Firestore
    att_docs = db.collection("attendance").where("name", "==", name).get()
    for i in range(0, len(att_docs), 500):
        batch = db.batch()
        for doc in att_docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()
    _remove_person_from_meta(name)

    return redirect("/faces")

# ================= ADD FACE (single) =================
@app.route("/add_face", methods=["GET", "POST"])
@login_required
def add_face():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        file = request.files.get("image")

        if not name or not file:
            return "Name or image missing"

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        enc = face_recognition.face_encodings(np.array(img_pil))
        if not enc:
            return "No face detected. Use a clear photo."

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        img_pil.save(os.path.join(person_dir, "profile.jpg"), "JPEG")

        known_encodings.append(enc[0])
        known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        return redirect("/faces")

    return render_template("add_face.html")

# ================= ADD FACE UPLOAD (auto 10) =================
@app.route("/add_face_upload", methods=["GET", "POST"])
@login_required
def add_face_upload():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        file = request.files.get("image")

        if not name or not file:
            return "Name or image missing"

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        for i in range(1, 11):
            img_path = os.path.join(person_dir, f"img_{i}.jpg")
            img_pil.save(img_path, "JPEG")
            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        return redirect("/faces")

    return render_template("add_face_upload.html")

# ================= ADD FACE CAPTURE (10 files upload) =================
@app.route("/add_face_capture", methods=["GET", "POST"])
@login_required
def add_face_capture():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        files = request.files.getlist("images")

        if not name or len(files) != 10:
            return "10 images required"

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        for i, file in enumerate(files, start=1):
            img_pil = Image.open(file.stream).convert("RGB")
            img_pil.thumbnail((640, 480))

            img_path = os.path.join(person_dir, f"img_{i}.jpg")
            img_pil.save(img_path, "JPEG")

            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        return redirect("/faces")

    return render_template("add_face_capture.html")

# ================= ATTENDANCE =================
@app.route("/attendance")
@login_required
def attendance():
    meta_doc = db.collection("metadata").document("attendance_persons").get()
    persons = meta_doc.to_dict().get("names", []) if meta_doc.exists else []
    return render_template("attendance_list.html", persons=persons)

@app.route("/attendance/<name>")
@login_required
def attendance_person(name):
    docs = (
        db.collection("attendance")
        .where("name", "==", name)
        .order_by("date")
        .order_by("time")
        .get()
    )

    grouped = OrderedDict()
    for doc in docs:
        d = doc.to_dict()
        date = d["date"]
        if date not in grouped:
            grouped[date] = []
        grouped[date].append(d)

    records = []
    for date, entries in grouped.items():
        in_time, out_time = None, None
        for e in entries:
            if e["status"] == "IN" and not in_time:
                in_time = e["time"]
            if e["status"] == "OUT":
                out_time = e["time"]

        work_duration = "--"
        if in_time and out_time:
            t1 = datetime.strptime(in_time, "%H:%M:%S")
            t2 = datetime.strptime(out_time, "%H:%M:%S")
            diff = t2 - t1
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            work_duration = f"{hours:02d}:{minutes:02d}"

        records.append({
            "Date": date,
            "In": in_time or "--",
            "Out": out_time or "--",
            "Work": work_duration,
        })

    return render_template("attendance_person.html", name=name, records=records)

# ================= DELETE BY DATE =================
@app.route("/attendance/delete_by_date", methods=["POST"])
@login_required
def delete_attendance_by_date():
    name = request.form.get("name")
    from_date = request.form.get("from_date")
    to_date = request.form.get("to_date")

    docs = (
        db.collection("attendance")
        .where("name", "==", name)
        .where("date", ">=", from_date)
        .where("date", "<=", to_date)
        .get()
    )

    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()

    # Remove person from metadata if no records remain
    remaining = db.collection("attendance").where("name", "==", name).limit(1).get()
    if not remaining:
        _remove_person_from_meta(name)

    return redirect(f"/attendance/{name}")

# ================= DELETE PERSON ATTENDANCE =================
@app.route("/attendance/delete_person/<name>")
@login_required
def delete_person_attendance(name):
    docs = db.collection("attendance").where("name", "==", name).get()
    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()

    _remove_person_from_meta(name)
    return redirect("/attendance")

# ================= EXPORT =================
@app.route("/export/<name>")
@login_required
def export_person(name):
    docs = (
        db.collection("attendance")
        .where("name", "==", name)
        .order_by("date")
        .order_by("time")
        .get()
    )

    if not docs:
        return "No attendance data found"

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Time", "Name", "Status"])
    for doc in docs:
        d = doc.to_dict()
        writer.writerow([d["date"], d["time"], d["name"], d["status"]])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        as_attachment=True,
        download_name=f"{name}_attendance.csv",
        mimetype="text/csv",
    )

# ================= CHARTS =================
@app.route("/charts")
@login_required
def charts():
    docs = db.collection("attendance").where("status", "==", "IN").get()

    counts = {}
    for doc in docs:
        n = doc.to_dict()["name"]
        counts[n] = counts.get(n, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())

    plt.clf()
    if labels:
        plt.bar(labels, values)
    plt.title("Daily IN Count")
    plt.savefig("static/chart.png")

    return render_template("charts.html")

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

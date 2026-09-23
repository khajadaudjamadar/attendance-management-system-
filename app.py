"""
Attendance Management System
Flask + OpenCV (Haar Cascade detection + LBPH recognition) + SQLite
"""
import os
import json
import base64
import sqlite3
from datetime import datetime, date

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
FACES_DIR = os.path.join(BASE_DIR, "static", "faces")
MODEL_PATH = os.path.join(BASE_DIR, "database", "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "database", "labels.json")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLES_PER_PERSON = 20          # face images captured during registration
FACE_SIZE = (200, 200)           # normalized size for training/recognition
RECOGNITION_THRESHOLD = 70.0     # LBPH confidence: LOWER = more confident match

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'employee')),
            identifier TEXT NOT NULL,      -- roll no / employee id
            department TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            att_date TEXT NOT NULL,
            att_time TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY(person_id) REFERENCES people(id),
            UNIQUE(person_id, att_date)
        );
        """
    )
    conn.commit()
    conn.close()


def load_model_if_exists():
    """Load a previously trained LBPH model, if one exists."""
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        recognizer.read(MODEL_PATH)
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


LABELS = load_model_if_exists()  # {str(label_id): person_id}


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def decode_base64_image(data_url):
    """Convert a 'data:image/jpeg;base64,...' string from the browser into a cv2 BGR image."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_largest_face(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick the largest detected face (closest to camera)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]  # (x, y, w, h)


def extract_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    box = detect_largest_face(gray)
    if box is None:
        return None
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)  # normalize lighting
    return face


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_model():
    """Rebuild the LBPH model from every saved face image in static/faces/<person_id>/."""
    faces, labels, label_map = [], [], {}
    next_label = 0

    for person_dir in sorted(os.listdir(FACES_DIR)):
        person_path = os.path.join(FACES_DIR, person_dir)
        if not os.path.isdir(person_path):
            continue
        try:
            person_id = int(person_dir)
        except ValueError:
            continue

        label_map[str(next_label)] = person_id
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(person_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, FACE_SIZE))
            labels.append(next_label)
        next_label += 1

    if not faces:
        return False, "No face data to train on yet."

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    global LABELS
    LABELS = label_map
    return True, f"Model trained on {len(faces)} images across {len(label_map)} people."


# --------------------------------------------------------------------------
# Routes: pages
# --------------------------------------------------------------------------
@app.route("/")
def home():
    conn = get_db()
    total_people = conn.execute("SELECT COUNT(*) c FROM people").fetchone()["c"]
    today_count = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE att_date = ?", (str(date.today()),)
    ).fetchone()["c"]
    conn.close()
    return render_template("index.html", total_people=total_people, today_count=today_count)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        role = request.form["role"]
        identifier = request.form["identifier"].strip()
        department = request.form.get("department", "").strip()

        conn = get_db()
        cur = conn.execute(
            "INSERT INTO people (name, role, identifier, department, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, role, identifier, department, datetime.now().isoformat()),
        )
        person_id = cur.lastrowid
        conn.commit()
        conn.close()

        os.makedirs(os.path.join(FACES_DIR, str(person_id)), exist_ok=True)
        return redirect(url_for("capture_faces", person_id=person_id))

    return render_template("register.html")


@app.route("/capture/<int:person_id>")
def capture_faces(person_id):
    conn = get_db()
    person = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    if person is None:
        return redirect(url_for("register"))
    return render_template("capture.html", person=person, samples_needed=SAMPLES_PER_PERSON)


@app.route("/mark")
def mark_attendance_page():
    return render_template("mark.html")


@app.route("/dashboard")
def dashboard():
    conn = get_db()
    people = conn.execute("SELECT * FROM people ORDER BY name").fetchall()
    today_records = conn.execute(
        """SELECT attendance.*, people.name, people.role, people.identifier
           FROM attendance JOIN people ON attendance.person_id = people.id
           WHERE att_date = ? ORDER BY att_time DESC""",
        (str(date.today()),),
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", people=people, records=today_records, today=date.today())


# --------------------------------------------------------------------------
# Routes: API (called via JS from the browser)
# --------------------------------------------------------------------------
@app.route("/api/capture_frame", methods=["POST"])
def api_capture_frame():
    """Receive one webcam frame during registration, save the cropped face."""
    data = request.get_json()
    person_id = data["person_id"]
    frame_idx = data["frame_idx"]
    img = decode_base64_image(data["image"])

    face = extract_face(img)
    if face is None:
        return jsonify({"ok": False, "message": "No face detected. Center your face in the frame."})

    save_path = os.path.join(FACES_DIR, str(person_id), f"{frame_idx}.jpg")
    cv2.imwrite(save_path, face)
    return jsonify({"ok": True, "message": f"Captured {frame_idx + 1}/{SAMPLES_PER_PERSON}"})


@app.route("/api/train", methods=["POST"])
def api_train():
    ok, message = train_model()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Receive one webcam frame, identify the person, and log attendance if not already marked today."""
    if not LABELS:
        return jsonify({"ok": False, "message": "No trained model yet. Register people and train first."})

    data = request.get_json()
    img = decode_base64_image(data["image"])
    face = extract_face(img)
    if face is None:
        return jsonify({"ok": False, "message": "No face detected."})

    label, confidence = recognizer.predict(face)  # lower confidence = better match
    person_id = LABELS.get(str(label))
    if person_id is None or confidence > RECOGNITION_THRESHOLD:
        return jsonify({"ok": False, "message": "Face not recognized."})

    conn = get_db()
    person = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
    if person is None:
        conn.close()
        return jsonify({"ok": False, "message": "Matched person no longer exists."})

    today = str(date.today())
    already = conn.execute(
        "SELECT 1 FROM attendance WHERE person_id = ? AND att_date = ?", (person_id, today)
    ).fetchone()

    if already:
        conn.close()
        return jsonify({
            "ok": True, "already_marked": True,
            "name": person["name"], "message": f"{person['name']} already marked present today."
        })

    conn.execute(
        "INSERT INTO attendance (person_id, att_date, att_time, confidence) VALUES (?, ?, ?, ?)",
        (person_id, today, datetime.now().strftime("%H:%M:%S"), float(confidence)),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "ok": True, "already_marked": False,
        "name": person["name"], "role": person["role"],
        "message": f"Attendance marked for {person['name']}"
    })


@app.route("/api/delete_person/<int:person_id>", methods=["POST"])
def api_delete_person(person_id):
    conn = get_db()
    conn.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()

    person_dir = os.path.join(FACES_DIR, str(person_id))
    if os.path.isdir(person_dir):
        for f in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, f))
        os.rmdir(person_dir)

    train_model()  # retrain without this person
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)

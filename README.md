# AttendX — Attendance Management System

A Flask web app that marks attendance for students and employees using
face recognition (OpenCV Haar Cascade for detection + LBPH for recognition —
no heavy dependencies like `dlib`, so it installs cleanly everywhere).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in a browser. The browser (not the server)
accesses your webcam, so it must be opened over `localhost` or `https` —
browsers block camera access on plain `http://<ip>` for other devices.

## How it works

1. **Register** — add a person (name, role, roll no. / employee ID, department).
2. **Capture Face** — the browser streams webcam frames to the server, which
   detects and saves 20 cropped, normalized face samples for that person.
3. **Train** — happens automatically after capture (`/api/train`), building an
   LBPH model from every registered person's saved face images.
4. **Mark Attendance** — on the Mark Attendance page, the browser sends a frame
   every 2 seconds; the server detects a face, matches it against the trained
   model, and logs one attendance record per person per day.
5. **Dashboard** — view today's attendance log and manage registered people.

## Project structure

```
attendance-system/
├── app.py                  # Flask app: routes + face detection/recognition
├── requirements.txt
├── database/
│   ├── attendance.db       # SQLite (created on first run)
│   ├── face_model.yml      # trained LBPH model (created after first training)
│   └── labels.json         # maps model label -> person_id
├── static/
│   ├── css/style.css
│   └── faces/<person_id>/  # saved face samples per person
└── templates/
    ├── base.html, index.html, register.html
    ├── capture.html, mark.html, dashboard.html
```

## Tuning recognition

In `app.py`:
- `RECOGNITION_THRESHOLD` (default `70.0`) — LBPH confidence score; **lower
  is a stricter/better match**. If strangers get matched, lower this. If real
  people aren't recognized, raise it slightly.
- `SAMPLES_PER_PERSON` (default `20`) — more samples improve accuracy but
  make registration slower.

## Known limitations (fine for a fresher/portfolio project, worth knowing)

- LBPH is lighter than deep-learning face recognition (e.g. `face_recognition`/
  dlib or FaceNet) and less accurate under poor lighting or with lookalikes.
  It was chosen here because it installs with plain `pip` and needs no
  compiled dependencies.
- One face model file is shared across everyone (no per-department isolation).
- No authentication/admin login yet — anyone with network access can register
  people or view the dashboard.
- No liveness check — a photo held up to the camera could fool it. For a real
  deployment, add blink detection or an IR/depth camera.

## Suggested next steps

- Add an admin login (Flask-Login) before exposing this beyond localhost.
- Add CSV/Excel export for attendance records (pairs well with your existing
  Pandas/Excel experience).
- Add a date-range filter and per-person attendance history on the dashboard.
- Swap LBPH for `face_recognition` (dlib) or a deep model if accuracy needs
  to improve and you can install compiled dependencies.

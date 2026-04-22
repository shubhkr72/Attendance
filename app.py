from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import requests
import time
from collections import defaultdict

app = Flask(__name__)

# ---------- CONFIG ----------
KNOWN_DIR = "Images"
ATTENDANCE_CSV = "attendance.csv"
TOLERANCE = 0.50
EAR_THRESH = 0.22
CONSEC_FRAMES_CLOSED = 2
BLINK_WINDOW_SEC = 1.5
CAM_INDEX = 0
# ----------------------------

# Load known faces
def load_known_encodings(known_dir):
    encodings, names = [], []
    for fname in os.listdir(known_dir):
        path = os.path.join(known_dir, fname)
        if not os.path.isfile(path): continue
        name, ext = os.path.splitext(fname)
        try:
            img = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(img)
            if len(boxes) == 0: continue
            enc = face_recognition.face_encodings(img, boxes)[0]
            encodings.append(enc)
            names.append(name)
            print(f"[enrolled] {name}")
        except Exception as e:
            print("error loading", fname, e)
    return encodings, names

known_encodings, known_names = load_known_encodings(KNOWN_DIR)
state = defaultdict(lambda: {"closed_count":0, "blinked":False, "last_marked":0})

# ----------- LOCATION ----------
def get_location():
    try:
        r = requests.get("https://ipinfo.io", timeout=3)
        loc = r.json().get("loc", "0,0")
        lat, lon = loc.split(",")
        return lat, lon
    except Exception:
        return None, None

# ----------- ATTENDANCE ----------
def mark_attendance(name):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lat, lon = get_location()
    df = pd.DataFrame([[name, ts, lat, lon]], columns=["name", "timestamp", "latitude", "longitude"])
    try:
        if not os.path.exists(ATTENDANCE_CSV):
            df.to_csv(ATTENDANCE_CSV, index=False, mode='w', header=True)
        else:
            df.to_csv(ATTENDANCE_CSV, index=False, mode='a', header=False)
        print(f"[marked] {name} at {ts} ({lat}, {lon})")
    except PermissionError:
        print("[WARN] attendance.csv is locked. Retry later.")

# ----------- BLINK HELPER ----------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# ----------- VIDEO STREAM ----------
def gen_frames():
    video = cv2.VideoCapture(CAM_INDEX)
    while True:
        success, frame = video.read()
        if not success:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, boxes)
        now = time.time()

        for (box, enc) in zip(boxes, encs):
            top, right, bottom, left = [v*2 for v in box]
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            name = "Unknown"
            if True in matches:
                best_idx = np.argmin(face_distances)
                name = known_names[best_idx]

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if name != "Unknown":
                landmarks_list = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])
                if landmarks_list:
                    landmarks = landmarks_list[0]
                    left_eye = landmarks.get("left_eye", [])
                    right_eye = landmarks.get("right_eye", [])
                    if len(left_eye) >= 6 and len(right_eye) >= 6:
                        left_EAR = eye_aspect_ratio(left_eye)
                        right_EAR = eye_aspect_ratio(right_eye)
                        ear = (left_EAR + right_EAR) / 2.0
                        cv2.putText(frame, f"EAR:{ear:.2f}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        st = state[name]
                        if ear < EAR_THRESH:
                            st["closed_count"] += 1
                        else:
                            if st["closed_count"] >= CONSEC_FRAMES_CLOSED and not st["blinked"]:
                                mark_attendance(name)
                                st["blinked"] = True
                                st["last_marked"] = now
                            st["closed_count"] = 0
        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        return df.to_html(index=False)
    return "<p>No attendance records yet.</p>"

if __name__ == '__main__':
    app.run(debug=True)

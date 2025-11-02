import io
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, Response
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ---------------- CONFIGURATION ----------------
app = Flask(__name__)

MODEL_PATH = "face_emotionModel.h5"
DB_PATH = "database.db"

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load trained model
model = load_model(MODEL_PATH)

camera_running = False


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            emotion TEXT,
            image_path BLOB,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()


# ---------------- IMAGE EMOTION PREDICTION ----------------
def predict_emotion_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((48, 48))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    return EMOTION_LABELS[emotion_index]


# ---------------- FRAME EMOTION PREDICTION (LIVE) ----------------
def predict_emotion_from_frame(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_array = np.expand_dims(np.expand_dims(face_img, -1), 0) / 255.0

    prediction = model.predict(face_array)
    emotion_index = np.argmax(prediction)
    return EMOTION_LABELS[emotion_index]


# ---------------- VIDEO FEED ----------------
def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotion = predict_emotion_from_frame(face_roi)

            color = (0, 255, 0) if emotion == "happy" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    global camera_running
    message = ""
    show_camera = False
    if request.method == "POST":
        if "submit_image" in request.form:
            name = request.form.get("name")
            email = request.form.get("email")
            file = request.files.get("image")

            if not name or not email or not file:
                message = "⚠️ Please fill in all fields and upload an image."
            else:
                img_bytes = file.read()
                emotion = predict_emotion_from_bytes(img_bytes)

                # Save to DB directly as binary
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO students (name, email, emotion, image_path, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, email, emotion, img_bytes, datetime.now()))
                conn.commit()
                conn.close()

                message = f"{emotion}. " \
                        f"{'Why are you sad?' if emotion == 'sad' else 'Nice to see you smiling!' if emotion == 'happy' else ''}"
        elif "start_camera" in request.form:
            camera_running = True
            show_camera = True
        elif "stop_camera" in request.form:
            camera_running = False
            show_camera = False

    return render_template("index.html", message=message, show_camera=show_camera)


@app.route("/video_feed")
def video_feed():
    global camera_running
    if not camera_running:
        return Response(status=204)  # No content
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)



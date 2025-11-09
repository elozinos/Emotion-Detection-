from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import sqlite3
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# Load CNN Emotion Detection Model
# -----------------------------
model = load_model("face_emotionModel.h5")
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Database Setup
# -----------------------------
DATABASE = "users_results.db"

def init_db():
    """Initialize the SQLite database with required columns."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image_path TEXT,
            face_index INTEGER,
            emotion TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    name = request.form.get("name", "Anonymous")  # Optional name input

    # Save uploaded image
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    # Read image for processing
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Failed to read uploaded image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []

    for idx, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        prediction = model.predict(face_input)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        emotion = emotion_labels.get(class_idx, "Unknown")

        results.append({
            "face_index": idx,
            "name": name,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

        # Save to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute(
            "INSERT INTO results (name, image_path, face_index, emotion, confidence) VALUES (?, ?, ?, ?, ?)",
            (name, image_path, idx, emotion, confidence)
        )
        conn.commit()
        conn.close()

    return jsonify(results)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load face model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trained_model.xml")  # Make sure this file exists

# Haarcascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Label decoder
names = {0: "YourName"}  # Change this to match your dataset labels

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["image"]

    # Convert image to array
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face)
        results.append({
            "name": names.get(label, "Unknown"),
            "confidence": round(confidence, 2)
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


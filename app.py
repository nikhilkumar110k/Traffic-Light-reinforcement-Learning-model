from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

model = tf.keras.models.load_model("saved_model/traffic_light_agent.h5")
colors = {
    0: (0, 0, 255),  
    1: (0, 255, 255),  
    2: (0, 255, 0)  
}

car_cascade_path = './cascades/haarcascade_car.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_path)

def lane_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    roi = edges[int(height / 2):height, 0:width]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines, edges

def get_traffic_light_action(traffic_volume, state_size):
    state = np.array([traffic_volume] * state_size)
    state = np.reshape(state, [1, state_size])
    action_values = model.predict(state)
    action = np.argmax(action_values[0])
    return action

def detect_and_label_cars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for idx, (x, y, w, h) in enumerate(cars, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Car {idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, len(cars)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded.", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file.", 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return Response(process_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def use_camera():
    return Response(process_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def process_camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def process_frame(frame):
    height, width, _ = frame.shape
    lines, edges = lane_detection(frame)

    frame, car_count = detect_and_label_cars(frame)

    traffic_volume = car_count
    action = get_traffic_light_action(traffic_volume, 2)
    color = colors[action]

    cv2.rectangle(frame, (100, 100), (200, 300), color, -1)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, int(y1 + height / 2)), (x2, int(y2 + height / 2)), (255, 0, 0), 2)

    return frame

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

model = tf.keras.models.load_model("saved_model/4lanestraffic_light_agent.h5")
state_size = model.input_shape[1]
colors = {
    0: (0, 0, 255),  
    1: (0, 255, 255),  
    2: (0, 255, 0)  
}

car_cascade_path = './cascades/haarcascade_car.xml'
if not os.path.exists(car_cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at: {car_cascade_path}")

car_cascade = cv2.CascadeClassifier(car_cascade_path)

def generate_co2_emissions(traffic_volume):
    return traffic_volume * 400

def lane_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    roi = edges[int(height / 2):height, 0:width]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines, edges

def get_traffic_light_action(traffic_volume):
    if isinstance(traffic_volume, int):
        traffic_volume = [traffic_volume]
    while len(traffic_volume) < state_size:
        traffic_volume.extend(traffic_volume)  
    traffic_volume = traffic_volume[:state_size]  

    state = np.array(traffic_volume)
    state = np.reshape(state, [1, state_size])  
    action_values = model.predict(state, verbose=0)
    action = np.argmax(action_values[0])
    return action

def detect_and_label_cars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    return len(cars)

@app.route('/')
def index():
    return render_template('4lanesindex.html')

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
    lane_splits = width // 4

    green_count = 0  
    metrics = {}

    for i in range(4):
        lane = frame[:, i * lane_splits:(i + 1) * lane_splits]
        car_count = detect_and_label_cars(lane)
        co2_emissions = generate_co2_emissions(car_count)
        action = get_traffic_light_action([car_count])

        if action == 2: 
            if green_count < 2:
                green_count += 1
            else:
                action = 0  

        metrics[f"lane_{i + 1}"] = {"car_count": car_count, "co2": co2_emissions, "action": action}

    for i, (lane, data) in enumerate(metrics.items()):
        x = i * lane_splits
        color = colors[data["action"]]
        cv2.rectangle(frame, (x, 0), (x + lane_splits, height), color, 2)
        overlay_text = f"{lane}: Cars={data['car_count']} | CO2={data['co2']:.2f}g"
        cv2.putText(frame, overlay_text, (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False)

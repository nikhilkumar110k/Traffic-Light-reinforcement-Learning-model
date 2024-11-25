from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import tensorflow as tf
import os
import random

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

def get_traffic_light_action(traffic_volume, current_green_count):
    
    if isinstance(traffic_volume, int):
        traffic_volume = [traffic_volume]
    while len(traffic_volume) < state_size:
        traffic_volume.extend(traffic_volume)  
    traffic_volume = traffic_volume[:state_size]

    state = np.array(traffic_volume).reshape(1, -1)
    action_values = model.predict(state, verbose=0)
    action = np.argmax(action_values[0])

    if action == 2 and current_green_count >= 2:
        action = 0  
    return action


def detect_and_label_cars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    return len(cars)

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

import random

def process_frame(frame):
    
    height, width, _ = frame.shape
    lane_splits = width  

    green_count = 0  
    lane_signals = [] 
    metrics_text = []  

    main_lane_cars = detect_and_label_cars(frame)
    main_co2_emissions = generate_co2_emissions(main_lane_cars)
    main_action = get_traffic_light_action(main_lane_cars, green_count)
    if main_action == 2:  
        green_count += 1
    lane_signals.append(main_action)

    metrics_text.append((
        f"Main Lane: Cars={main_lane_cars} | CO2={main_co2_emissions:.2f} | Signal={['Red', 'Yellow', 'Green'][main_action]}",
        colors[main_action]
    ))

    
    for i in range(1, 4): 
        random_car_count = random.randint(5, 20) 
        random_co2_emissions = generate_co2_emissions(random_car_count)

        if green_count < 2:
            random_action = get_traffic_light_action(random_car_count, green_count)
            if random_action == 2:  
                green_count += 1
        else:
            random_action = random.choice([0, 1]) 

        lane_signals.append(random_action)

        metrics_text.append((
            f"Lane {i}: Cars={random_car_count} | CO2={random_co2_emissions:.2f} | Signal={['Red', 'Yellow', 'Green'][random_action]}",
            colors[random_action]
        ))

    y_position = 30
    for line, color in metrics_text:
        cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_position += 30

    return frame




if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False)

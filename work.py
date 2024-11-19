import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("saved_model/traffic_light_agent.h5")

colors = {
    0: (0, 0, 255), 
    1: (0, 255, 255),  
    2: (0, 255, 0)  
}

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


def generate_synthetic_traffic_data(num_samples=1000):
    times_of_day = np.random.uniform(0, 24, num_samples)  
    traffic_volumes = np.random.randint(0, 101, num_samples)  
    return np.column_stack((times_of_day, traffic_volumes))

data = generate_synthetic_traffic_data(num_samples=1000)
state_size = data.shape[1] 


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape  
    lines, edges = lane_detection(frame)
    
    traffic_volume = len(lines) if lines is not None else 0
    
    action = get_traffic_light_action(traffic_volume, state_size)
    
    if action == 0: 
        cv2.rectangle(frame, (100, 100), (200, 300), colors[0], -1)
    elif action == 1: 
        cv2.rectangle(frame, (100, 100), (200, 300), colors[1], -1)
    elif action == 2: 
        cv2.rectangle(frame, (100, 100), (200, 300), colors[2], -1)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, int(y1 + height / 2)), (x2, int(y2 + height / 2)), (255, 0, 0), 2)

    cv2.imshow('Edges', edges)

    cv2.imshow('Traffic Light Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

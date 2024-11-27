from flask import Flask, Response, render_template, request, jsonify
from picamera2 import Picamera2
import cv2
import RPi.GPIO as GPIO
import time
import torch
from ultralytics import YOLO

app = Flask(__name__)
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

yolo_model = YOLO('bestx.pt')  # Replace with your YOLO model path
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')  # Use 'cuda' if a GPU is available
midas.eval()
# Load MiDaS transformation
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform
# Pin Definitions for GPIO
in1 = 24
in2 = 23
en = 25
in3 = 16
in4 = 20
enx = 21

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(enx, GPIO.OUT)

# Initialize PWM
p = GPIO.PWM(en, 1000)
p.start(50)
q = GPIO.PWM(enx, 1000)
q.start(50)

def init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(in3, GPIO.OUT)
    GPIO.setup(in4, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)
    GPIO.setup(enx, GPIO.OUT)

def forward():
    print("Motion Forward")
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    time.sleep(0.2)
    stop()

def backward():
    print("Motion Backward")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    time.sleep(0.2)
    stop()

def left():
    print("Motion Left")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    time.sleep(0.2)
    stop()

def right():
    print("Motion Right")
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    time.sleep(0.2)
    stop()

def stop():
    print("Motion Stop")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

# Initialize GPIO
MIN_CONTOUR_AREA = 500
init()
stop()


def Colorate(frame, datas):
    dangerzones = []
    left = (0, 162)
    center = (162, 477)
    right = (477, 640)

    overlay = frame.copy()  # Create a copy of the frame for overlay

    for data in datas:
        if left[0] <= data < left[1]:
            print("It's in Left")
            overlay[:, left[0]:left[1]] = (0, 0, 255)  # Add red overlay to the left
            dangerzones.append("left")
        elif center[0] <= data < center[1]:
            print("It's in Centre")
            overlay[:, :] = (0, 0, 255)  # Add red overlay to the entire frame
            dangerzones.extend(["left", "right", "forward"])
            break  # Once in the center, the entire frame is marked red, so exit
        elif right[0] <= data < right[1]:
            print("It's in Right")
            overlay[:, right[0]:right[1]] = (0, 0, 255)  # Add red overlay to the right
            dangerzones.append("right")

    # Blend the original image with the overlay to create transparency
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Return the frame and unique danger zones
    return frame, list(set(dangerzones))
    
def Coloratelite(frame, datas):
    dangerzones = []
    left = (0, 162)
    center = (162, 477)
    right = (477, 640)

    overlay = frame.copy()  # Create a copy of the frame for overlay

    for data in datas:
        if left[0] <= data < left[1]:
            print("It's in Left")
            overlay[:, left[0]:left[1]] = (0, 0, 255)  # Add red overlay to the left
            dangerzones.append("left")
        elif center[0] <= data < center[1]:
            print("It's in Centre")
            overlay[:, center[0]:center[1]] = (0, 0, 255)  # Add red overlay to the entire frame
            dangerzones.extend(["forward"])
            break  # Once in the center, the entire frame is marked red, so exit
        elif right[0] <= data < right[1]:
            print("It's in Right")
            overlay[:, right[0]:right[1]] = (0, 0, 255)  # Add red overlay to the right
            dangerzones.append("right")

    # Blend the original image with the overlay to create transparency
    alpha = 0.15  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Return the frame and unique danger zones
    return frame, list(set(dangerzones))

globe="stop"
def generate_frames():
    global globe
    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        C,Dir=PreLane(frame)
        detections,checks1,checks2 = pfd(
            frame, yolo_model, midas, transform, device='cpu'
        )

        frame=annotate_frame(frame,detections)
        frame,danger1=Colorate(frame,checks1)
        frame,danger2=Coloratelite(frame,checks2)
        dangers=list(set(danger1+danger2))
        if Dir not in dangers:
            globe=Dir
            cv2.putText(frame, f"Centre {C} Dir {Dir}",(20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            Dir="stop"
            cv2.putText(frame, f"Centre {C} Dir Stop",(20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def PreLane(frame):
    frame = frame[300:500, :]
    cX=None
    cY=None
    Dir=None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            #cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if cX < 300:
                    Dir="left"
                elif cX > 400:
                    Dir="right"
                else:
                    Dir="forward"
    return ((cX,cY),Dir)

def annotate_frame(frame, detections):
    for detection in detections:
        label, confidence, box, center, depth = detection
        x1, y1, x2, y2 = box
        center_x, center_y = center

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Obstacle ({confidence:.2f}) Depth: {depth:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def pfd(frame, model, midas, transform, device='cpu'):
    results = model(frame)
    input_batch = transform(frame).to(device)

    with torch.no_grad():
        depth_prediction = midas(input_batch)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        depth_map = depth_prediction.cpu().numpy()

    detections = []
    check1=[]
    check2=[]
    
    for box in results[0].boxes:
        confidence = float(box.conf)
        if confidence>=0.80:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth_value = depth_map[center_y, center_x]
            if depth_value>550:
                check1.append(center_x)
                check2.append(x1)
                check2.append(x2)
            label_index = int(box.cls)
            confidence = float(box.conf)
            label = results[0].names[label_index]

            detection = [label, confidence, [x1, y1, x2, y2], [center_x, center_y], depth_value]
            detections.append(detection)
        
    return detections,check1,check2

@app.route('/videofeed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')
def move(action):
    if action=="forward":
        forward()
    elif action=="backward":
        backward()
    elif action=="left":
        left()
    elif action=="right":
        right()
    elif action=="stop":
        stop()
@app.route('/control', methods=['POST'])
def control():
    action = request.json.get('action')
    print(f"Action received: {action}")
    if action!="auto":
        move(action)
    else:
        move(globe)
    return jsonify({'status': 'success', 'action': action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

GPIO.cleanup()

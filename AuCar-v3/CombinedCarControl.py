import RPi.GPIO as GPIO
import time
import cv2
import torch
import numpy as np
from picamera2 import Picamera2

# Pin Definitions
in1 = 24
in2 = 23
en = 25
in3 = 16
in4 = 20
enx = 21

# Output Mode GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(enx, GPIO.OUT)

# Initializing motors to low
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

# Speed Setting
p = GPIO.PWM(en, 1000)
p.start(40)
q = GPIO.PWM(enx, 1000)
q.start(40)

print("\n")


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


def backward():
    print("Motion Backward")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)


def left():
    print("Motion Left")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)


def right():
    print("Motion Right")
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)


def stop():
    print("Motion NULL")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)


# Load MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

# Start the camera
picam2.start()

# Before starting the motors
GPIO.cleanup()
init()

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame to RGB format and rotate it
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Transform the image for MiDaS
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Get the maximum pixel value from the depth output
    max_pixel_value = int(output.max())

    # Normalize the output for display
    output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
    output = (output * 255).astype('uint8')  # Scale to [0, 255]

    # Convert to a color map
    output_color = cv2.applyColorMap(output, cv2.COLORMAP_PLASMA)

    # Display the maximum pixel value on the depth map
    cv2.putText(output_color, f'Max Pixel Value: {max_pixel_value:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the depth map
    cv2.imshow('Depth Map', output_color)

    # Control the motors based on depth
    if max_pixel_value >= 800:
        #Do Something
        stop()
    else:
        #Do Something
        forward()
        time.sleep(0.2)
        stop()

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After Ending Stuff
picam2.stop()
cv2.destroyAllWindows()
GPIO.cleanup()





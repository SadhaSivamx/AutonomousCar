from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

# Start the camera
picam2.start()

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
    print("Motion Stop")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

# Initialize GPIO
init()

# Define the minimum contour area threshold
MIN_CONTOUR_AREA = 500  # Adjust this value based on your needs

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Rotate the frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Crop the frame
    frame = frame[300:, :]

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if the largest contour meets the minimum area threshold
        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            # Draw the largest contour
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            # Compute the moments of the largest contour
            M = cv2.moments(largest_contour)

            # Compute the center of the contour
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw the center of the contour
                cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
                cv2.putText(frame, f"Center: ({cX}, {cY})", (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Control movement based on the center's x-coordinate
                if cX < 200:
                    left()
                    time.sleep(0.3)
                    stop()
                elif cX > 400:
                    right()
                    time.sleep(0.3)
                    stop()
                else:
                    forward()
                    time.sleep(0.3)
                    stop()
    
    # Display the frame
    cv2.imshow('Camera Stream', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
GPIO.cleanup()

import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# Pin Definitions
in1 = 24
in2 = 23
en = 25
in3 = 16
in4 = 20
enx = 21

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(enx, GPIO.OUT)

# Setting initial state
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
    print("Motion Stop")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)


# Before Starting the Motors
GPIO.cleanup()
init()

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

# Start the camera
picam2.start()


def detect_direction_and_draw(img):
    # Preprocess the image
    img_resized = cv.resize(img, (640, 480))
    img_cropped = img_resized[300:, :]  # Crop the image
    img_blurred = cv.blur(img_cropped, (3, 3))
    ret, img_thresh = cv.threshold(img_blurred, 150, 255, cv.THRESH_BINARY)
    kernel = np.ones((11, 11), dtype=np.uint8)
    img_eroded = cv.erode(img_thresh, kernel, iterations=1)

    # Detect edges using Canny
    edges = cv.Canny(img_eroded, 50, 150, apertureSize=3)

    # Apply Hough Transform
    lines = cv.HoughLines(edges, 1, np.pi / 180, 50)

    # Find the line with the maximum points
    best_line = None
    max_votes = 0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            votes = np.sum(edges == 255)
            if votes > max_votes:
                max_votes = votes
                best_line = line[0]

    # Draw the best line on the image
    img_color = cv.cvtColor(img_cropped, cv.COLOR_GRAY2BGR)
    if best_line is not None:
        rho, theta = best_line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
        theta_deg = np.degrees(theta)

        # Determine direction based on the angle
        if 0 <= theta_deg < 30:
            if x1 <= 300:
                direction = "Right"
                right()
            elif x1 >= 300:
                direction = "Left"
                left()
        elif 30 <= theta_deg <= 50:
            direction = "Straight"
            forward()
        elif 50 < theta_deg <= 80 or (0 <= theta_deg <30 and x1<=300):
            direction = "Right"
            right()
        elif 80 < theta_deg <= 90:
            direction = "Stop"
            stop()
        elif 90 < theta_deg <= 160 or (0 <= theta_deg <30 and x1>=300):
            direction = "Left"
            left()
        elif 160 < theta_deg <= 180:
            direction = "Stop"
            stop()
        else:
            direction = "Unknown"

        # Pause after each movement
        time.sleep(0.5)
        stop()

        # Annotate the angle and direction on the image
        cv.putText(img_color, f"Angle: {theta_deg:.2f} degrees", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(img_color, f"Direction: {direction}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert original image to color to match the processed image
    img_orig_color = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)

    # Resize the processed image to match the width of the original image
    img_color_resized = cv.resize(img_color, (img_orig_color.shape[1], img_color.shape[0]))

    # Concatenate the original and processed images vertically
    concatenated_img = np.vstack((img_orig_color, img_color_resized))

    return concatenated_img


while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame to grayscale for processing
    frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    frame_gray = cv.rotate(frame_gray, cv.ROTATE_180)

    try:
        # Process the frame and get the result
        img_result = detect_direction_and_draw(frame_gray)
        cv.imshow('Camera Stream', img_result)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

# Release resources
GPIO.cleanup()
picam2.stop()
cv.destroyAllWindows()

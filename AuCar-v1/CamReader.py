from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

# Start the camera
picam2.start()

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame to a format suitable for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Display the frame
    cv2.imshow('Camera Stream', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()

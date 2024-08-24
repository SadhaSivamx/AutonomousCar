import cv2
import os
import matplotlib.pyplot as plt
# Load your cascade classifier
cascade = cv2.CascadeClassifier()
# Function to detect objects and return the image with detections highlighted
def detect_and_display(image, classifier):
    img =image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect objects
    detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(detections) > 0:
        # Select the detection with the largest area
        top_detection = max(detections, key=lambda rect: rect[2] * rect[3])

        # Draw a rectangle around the top detection
        x, y, w, h = top_detection
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Put the text "Human" above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Human"
        font_scale = 3
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate the position for the text
        text_x = x + (w - text_size[0]) // 2
        text_y = y - 20  # 10 pixels above the rectangle

        # Put the text on the image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

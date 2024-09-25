
# Import dependencies
import cv2
import torch
import numpy as np
from picamera2 import Picamera2

# Download the MiDaS model
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

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
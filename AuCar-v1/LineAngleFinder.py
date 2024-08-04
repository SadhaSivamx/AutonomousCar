import cv2 as cv
import matplotlib.pyplot as plt
import  numpy as np
def AngleFinder(img):
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
        cv.line(img_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
        theta_deg = np.degrees(theta)
        # Pause after each movement
        # Annotate the angle and direction on the image
        cv.putText(img_color, f"Angle: {theta_deg:.2f} degrees", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv.putText(img_color, f"Direction: {direction}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert original image to color to match the processed image
    img_orig_color = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)

    # Resize the processed image to match the width of the original image
    img_color_resized = cv.resize(img_color, (img_orig_color.shape[1], img_color.shape[0]))

    # Concatenate the original and processed images vertically
    concatenated_img = img_color_resized

    return concatenated_img

img=cv.imread("Src/11.png",0)
plt.imshow(AngleFinder(img))
plt.axis('off')
plt.show()
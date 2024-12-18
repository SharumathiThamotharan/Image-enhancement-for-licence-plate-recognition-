# Image-enhancement-for-licence-plate-recognition-
import cv2
import numpy as np

# Load the image
input_image_path = "license_plate.jpg"  # Replace with the path to your image
image = cv2.imread(input_image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection (Canny Edge Detection)
edges = cv2.Canny(blurred, 50, 150)

# Enhance contrast using Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Combine edges and equalized image for better enhancement
enhanced = cv2.addWeighted(edges, 0.5, equalized, 0.5, 0)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Enhanced Image", enhanced)

# Save the enhanced image
output_image_path = "enhanced_license_plate.jpg"
cv2.imwrite(output_image_path, enhanced)
print(f"Enhanced image saved as {output_image_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()

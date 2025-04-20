import cv2
import numpy as np

INPUT_IMAGE = "runtime/input/saved_drawing_original.png"
OUTPUT_IMAGE = "runtime/output/transparent_drawing.png"

image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError(f"Image not found: {INPUT_IMAGE}")

# Convert to grayscale for alpha mask
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create alpha: white background → transparent
_, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

# Combine BGR + alpha
b, g, r = cv2.split(image)
rgba = cv2.merge((b, g, r, alpha))

cv2.imwrite(OUTPUT_IMAGE, rgba)
print(f"[✔] Transparent drawing saved to: {OUTPUT_IMAGE}")
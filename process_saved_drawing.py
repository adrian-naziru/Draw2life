
import cv2
import numpy as np

INPUT_IMAGE = "runtime/input/saved_drawing_original.png"
OUTPUT_IMAGE = "runtime/output/transparent_drawing.png"

image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if image is None:
    raise FileNotFoundError(f"Image not found: {INPUT_IMAGE}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
b, g, r = cv2.split(image)
rgba = [b, g, r, alpha]
out = cv2.merge(rgba)

cv2.imwrite(OUTPUT_IMAGE, out)
print(f"[✔] Transparent drawing saved to: {OUTPUT_IMAGE}")

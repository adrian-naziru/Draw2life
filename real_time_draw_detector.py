import cv2 #pip3 install opencv-python
import numpy as np

# Load AR overlays (images with transparency)
overlays = {
    "Triangle": cv2.imread("data/shapes/triangle.png", cv2.IMREAD_UNCHANGED),
    "Square/Rectangle": cv2.imread("data/shapes/square.png", cv2.IMREAD_UNCHANGED),
    "Circle/Rounded shape": cv2.imread("data/shapes/circle.png", cv2.IMREAD_UNCHANGED)
}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh

def classify_and_get_contour(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "Nothing detected", None

    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.04 * cv2.arcLength(largest, True), True)
    corners = len(approx)

    if corners == 3:
        return "Triangle", largest
    elif corners == 4:
        return "Square/Rectangle", largest
    elif corners > 8:
        return "Circle/Rounded shape", largest
    else:
        return "Other shape", largest

def overlay_image_alpha(img, overlay, x, y):
    """ Overlay `overlay` onto `img` at (x, y) with alpha channel """
    h, w = overlay.shape[:2]

    if x + w > img.shape[1] or y + h > img.shape[0]:
        return img  # Don't overlay if it's out of bounds

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] +
                                alpha_l * img[y:y+h, x:x+w, c])
    return img

def main():
    cap = cv2.VideoCapture(1)  # adjust index if needed

    if not cap.isOpened():
        print("Could not access the webcam.")
        return

    confirmed_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thresh = preprocess_frame(frame)
        shape, contour = classify_and_get_contour(thresh)

        if shape in overlays and contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            resized_overlay = cv2.resize(overlays[shape], (w, h))
            frame = overlay_image_alpha(frame, resized_overlay, x, y)

        # Show instructions & detected shape
        cv2.putText(frame, f"Detected: {shape}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to confirm, 'q' to quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if confirmed_shape:
            cv2.putText(frame, f"Confirmed: {confirmed_shape}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

        cv2.imshow("AR Simulation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and shape != "Nothing detected":
            confirmed_shape = shape
            print(f"[✔] Shape confirmed: {confirmed_shape}")
            with open("confirmed_shape.txt", "w") as f:
                f.write(confirmed_shape)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

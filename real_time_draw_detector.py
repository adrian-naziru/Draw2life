
import cv2
import numpy as np

SHAPE_IMAGES = {
    "Triangle": cv2.imread("data/shapes/triangle.png", cv2.IMREAD_UNCHANGED),
    "Square/Rectangle": cv2.imread("data/shapes/square.png", cv2.IMREAD_UNCHANGED),
    "Circle/Rounded shape": cv2.imread("data/shapes/circle.png", cv2.IMREAD_UNCHANGED)
}

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh

def classify_shape(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
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

def overlay_image(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    if x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])
    return frame

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    confirmed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        thresh = preprocess(frame)
        shape, contour = classify_shape(thresh)

        if shape in SHAPE_IMAGES and contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            overlay = cv2.resize(SHAPE_IMAGES[shape], (w, h))
            frame = overlay_image(frame, overlay, x, y)

        cv2.putText(frame, f"Detected: {shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, "Press 'c' to confirm, 'q' to quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        if confirmed:
            cv2.putText(frame, f"Confirmed: {confirmed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,215,0), 2)

        cv2.imshow("Shape Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and shape != "Nothing detected":
            confirmed = shape
            # Save both the processed threshold image and the original full-color frame
            cv2.imwrite("runtime/input/saved_drawing.png", thresh)
            cv2.imwrite("runtime/input/saved_drawing_original.png", original_frame)
            with open("confirmed_shape.txt", "w") as f:
                f.write(shape)
            print("[✔] Saved both processed and original image.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

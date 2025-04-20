
import cv2
import numpy as np
import socket
import json
from ultralytics import YOLO

UDP_IP = "127.0.0.1"
UDP_PORT = 5051

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use 0 for internal cam, 1 for Iriun

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

overlay_img = cv2.imread("runtime/output/transparent_drawing.png", cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    raise FileNotFoundError("Missing transparent_drawing.png")

print("[🎥] Camera started. Sending data to 127.0.0.1:5051")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[0], conf=0.5, verbose=False)
    annotated_frame = frame.copy()

    frame_h, frame_w = frame.shape[:2]

    if results and results[0].boxes.data.shape[0] > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Resize overlay to max 10% of frame width, preserving aspect ratio
            target_w = int(frame_w * 0.10)
            aspect_ratio = overlay_img.shape[0] / overlay_img.shape[1]
            target_h = int(target_w * aspect_ratio)
            resized_overlay = cv2.resize(overlay_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Shoulder position
            shoulder_x = x1 + 10
            shoulder_y = y1 + int(0.25 * (y2 - y1))

            # Check bounds and apply overlay
            h, w = resized_overlay.shape[:2]
            if shoulder_y + h < frame_h and shoulder_x + w < frame_w:
                alpha_s = resized_overlay[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    annotated_frame[shoulder_y:shoulder_y + h, shoulder_x:shoulder_x + w, c] = (
                        alpha_s * resized_overlay[:, :, c] +
                        alpha_l * annotated_frame[shoulder_y:shoulder_y + h, shoulder_x:shoulder_x + w, c]
                    )
            else:
                print(f"[⚠️] Overlay out of bounds: shoulder=({shoulder_x},{shoulder_y}), size=({w}x{h}), frame=({frame_w}x{frame_h})")

            # Send UDP
            norm_x = cx / frame_w
            norm_y = cy / frame_h
            scale = (y2 - y1) / frame_h
            data = json.dumps({"x": norm_x, "y": norm_y, "scale": scale})
            sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO + Drawing on Shoulder", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

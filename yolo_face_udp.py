import cv2
from ultralytics import YOLO
import socket
import json

# Încarcă model YOLOv8 (nano = rapid)
model = YOLO("yolov8n.pt")

# Config UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5051
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Cameră de la Iriun
cap = cv2.VideoCapture(1)

print(f"[🎥] Camera pornită. Trimitem date către {UDP_IP}:{UDP_PORT}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = x2 - x1, y2 - y1

                # Normalizează poziția și dimensiunea
                frame_h, frame_w = frame.shape[:2]
                data = {
                    "x": round(cx / frame_w, 4),
                    "y": round(cy / frame_h, 4),
                    "scale": round(w / frame_w, 4)  # lățimea capului în %
                }

                sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

                # Vizualizare bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("YOLO Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
sock.close()

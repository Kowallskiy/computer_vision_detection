from ultralytics import YOLO
import cv2
import cvzone
import math

# cam = cv2.VideoCapture(0) # WebCamera
cam = cv2.VideoCapture("videos/cars.mp4")
cam.set(3, 640)
cam.set(4, 480)

model = YOLO("../yolo_weights/yolov8n.pt")

while True:
    success, img = cam.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            cls_name = "Human" if cls == 0 else "Object"

            cvzone.putTextRect(img, f"{cls_name} {conf}", (max(0, x1), max(35, y1)), scale=0.95, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
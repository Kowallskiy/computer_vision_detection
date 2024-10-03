
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort_tracker import Sort

# cam = cv2.VideoCapture(0) # WebCamera
cam = cv2.VideoCapture("videos/cars.mp4")
cam.set(3, 640)
cam.set(4, 480)

model = YOLO("../yolo_weights/yolov8n.pt")

mask = cv2.imread("images/mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cam.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=15)

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            cls_name = "Human" if cls == 0 else "Object"

            if cls in [2, 3, 5 ,7] and conf > 0.3:

                cvzone.putTextRect(img, f"{cls_name} {conf}", (max(0, x1), max(35, y1)), scale=0.95, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])

                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, ID = result

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
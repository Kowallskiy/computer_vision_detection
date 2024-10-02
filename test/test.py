from ultralytics import YOLO
import cv2

model = YOLO('../yolo_weights/yolov8n.pt')
result = model('images/cars.jpg', show=True)
cv2.waitKey(0)
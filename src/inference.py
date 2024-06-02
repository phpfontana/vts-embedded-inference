import cv2
from ultralytics import YOLO
from networks import YOLOv8
from time import time

# Initialize yolov8 object detector
model_sizes = ['n', 's', 'm', 'l', 'x']

for i in model_sizes:
    model_path = f"src/models/onnx/yolov8{i}.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

    # Read image
    img = cv2.imread("src/data/cars.jpg")

    # Detect Objects
    start = time()
    boxes, scores, class_ids = yolov8_detector(img)
    end = time()


    print(f"Model: Yolov8{model_sizes[i]}")
    print(f"Detection time: {end - start:.2f}s")

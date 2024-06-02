import cv2

from networks import YOLOv8
from time import time

# Initialize yolov8 object detector
model_path = "src/models/onnx/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
img = cv2.imread("src/data/cars.jpg")

# Detect Objects
start = time()
boxes, scores, class_ids = yolov8_detector(img)
end = time()

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
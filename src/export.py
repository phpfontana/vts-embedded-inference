from ultralytics import YOLO

model = YOLO('src/models/base/yolov8n.pt')

# Export to ONNX 
model.export(format='onnx')
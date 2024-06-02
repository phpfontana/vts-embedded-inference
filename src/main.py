from ultralytics import YOLO

model = YOLO('src/models/base/yolov9c.pt')

# Export to ONNX 
results = model.predict('src/data/cars.jpg')
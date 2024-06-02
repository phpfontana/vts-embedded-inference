from ultralytics import YOLO

model_sizes = ['n', 's', 'm', 'l', 'x']

for i in model_sizes:
    model = YOLO(f'src/models/onnx/yolov8{i}.pt')
    model.export(format='onnx', int8=True)

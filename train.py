from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(data="data.yaml", epochs=3)

results = model.val()

success = model.export(format="onnx")
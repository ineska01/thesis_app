from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model.train(data="coco90-seg.yaml", model="yolov8n-seg.pt", epochs=100, imgsz=640, batch=2, lr0=0.01)
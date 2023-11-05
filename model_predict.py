from ultralytics import YOLO

model = YOLO("best.pt")

model.predict('ort_102.png', save=True, imgsz=(1920,1088), conf=0.5)
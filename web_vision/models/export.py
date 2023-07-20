from ultralytics import YOLO
from detect import YOLO_WEIGHT


model = YOLO(YOLO_WEIGHT) 
model.export(format='engine', device=0)

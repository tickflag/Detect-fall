from ultralytics import YOLO
from settings import settings


model_pose = YOLO('yolov8s-pose.pt')

def yolo_pose(source):
    results = model_pose.predict(source=source, save=True, conf=0.3, vid_stride=1, show=True, device=0)
    return results

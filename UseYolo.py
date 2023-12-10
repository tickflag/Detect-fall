from ultralytics import YOLO
from settings import settings

model_pose = YOLO('./yolov8s-pose.pt')

def use_yolo(source):
    return model_pose.predict(
        source=source,
        save=False, #While debugging True, else False
        conf=0.5,
        vid_stride=1,
        device=settings['device'],
        show_labels=False,
        show_conf=False,
        verbose=False
    )
from catboost import CatBoostClassifier
import cv2
import datetime
from tkinter import messagebox
import threading

import UseYolo
from settings import settings



# Загружаем нашу модельку
tm = datetime.datetime.now()
model = CatBoostClassifier()
model.load_model('./cats-and-tea.json')
#print(datetime.datetime.now() - tm)


def process_frame(frame):
    cv2.imwrite('./input.jpg', frame)
    predict = check_pose('./input.jpg')

    return predict

def main():

    frames = []
    len_frames = 20

    cv2.namedWindow("video")
    vc = cv2.VideoCapture(settings['source'])
    
    if not vc.isOpened():
        print("Error: Couldn't open video source.")
        return 0
    
    while True:
        success, frame = vc.read()

        if not success:
            print("Error: Couldn't read frame.")
            break

        fallen_people = process_frame(frame)
        try:    
            fallen_people = max(fallen_people) // max(fallen_people)
        except Exception:
            fallen_people = 0
        frames.append(fallen_people)

        if len(frames) > len_frames:
            frames = frames[1::]
        
        print(frames)

        if sum(frames) / len_frames >= 0.9: #TODO Попробовать различные коэффиценты в случае наличия времени на одном датасете и выбрать лучшее
            print('человек упал\n', '-' * 50)
            x = threading.Thread(target=alarm_message)
            x.start()
            
            frames = []
            #TODO Написать функции запуска предупреждения через ткинтер в отдельном потоке или ченить другое использовать
    vc.release()



def check_pose(frame):
    time = datetime.datetime.now()
    result = UseYolo.use_yolo(frame)
    keypoints = []
    for res in result:
        for element in res:
            data = [0.0, 0.0]
            box = element.boxes.cpu().numpy().xyxy[0]
            for keypoint in element.keypoints.cpu().numpy().xy[0]:
                data[0] += round(int(keypoint[0]) / int(box[-1]), 3)
                data[1] += round(int(keypoint[1]) / int(box[2]), 3)
            keypoints.append(data)
    
    #print(f'Разметка и подготовка - {datetime.datetime.now() - time}')

    out_list = []

    for input_data in keypoints:
        metric = model.predict(prediction_type='Probability', data=input_data)
        out_list.append(0 if metric[0] >= metric[1] else 1)
        #print('Вероятность падения: ',metric[1])
    #print(datetime.datetime.now() - time)
    return out_list

def alarm_message():
    messagebox.showwarning(title='Внимание', message='Обнаружен упавший человек!')


if __name__ == '__main__':
    main()






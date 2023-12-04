import PredictPose
import settings
import cv2
import datetime

from tkinter import messagebox

def main(): #TODO write a func main that is going to обрабатывать кадр, \
    #todo считать кол-во лежащих людей(если они есть) и в случае если в течении 5 - 10 кадров подряд 
    #todo было более 1 упавшего человек, выдаем уведомление
    st = datetime.datetime.now()
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(settings.settings['source'])
    if vc.isOpened():
        success, frame = vc.read()
    else:
        success = False
    requared_frames = 5
    frames = 0
    lay_persons = 0
    while success:
        success, frame = vc.read()

        cv2.imwrite('input.jpg', frame)
        peoples = PredictPose.main('input.jpg')
        print(peoples)
        frames_to_add = 0
        for person in peoples:
            if person[0] >= 0.5:
                lay_persons += 1
                frames_to_add +=1
        if frames_to_add != 0: frames += 1
        else:   frames = 0
        if frames == requared_frames:
            #TODO write a code that shows an alarm that human is laying
            messagebox.showwarning("Alarm", "Обнаружен упавший человек")
            frames = 0
            break
        out = cv2.imread('output.jpg')
    
    print(datetime.datetime.now() - st)


if __name__ == "__main__":
    main()
import UseYolo 
import settings

import numpy as np
import math
import datetime

start_time = None

def main(source):
    global start_time
    start_time = datetime.datetime.now()
    peoples = []
    pose = UseYolo.yolo_pose(source=source)

    print(f"Затрачено на нейронку: {datetime.datetime.now() - start_time}")

    for res1 in pose:
        for element1 in res1:
            box = element1.boxes.cpu().numpy().xyxy[0]
            pose = element1.keypoints.cpu().numpy().xy[0]
            speed = xy_speed(box)
            chance = predict_possition_pose(pose=pose)
            peoples.append([chance, speed])
        
    return peoples

def xy_speed(xy): #TODO may do it as main algorithm with our predict-pose algorithm as second level \
    #todo it means we run algorithm that counts speed and if it thinks that person is fall, \
    #todo we use our predict-pose alg and from their results we predict if it is laying.
    return 0

def predict_possition_pose(pose):
    #global start_time

    left_shoulder = pose[5]
    right_shoulder = pose[6]
    left_hip = pose[11]
    right_hip = pose[12]
    
    head = pose[0]

    if [head[0], head[1]] == [0.0, 0.0]:
        for head_element in range(1, 4):
            if pose[head_element][0] != 0.0 and pose[head_element][1] != 0.0: 
                head = pose[head_element] 
                break
    
    angles = first_arg(left_shoulder=left_shoulder, left_hip=left_hip,
            right_shoulder=right_shoulder, right_hip=right_hip)
    btoa = second_arg(left_shoulder=left_shoulder, left_hip=left_hip, 
            right_shoulder=right_shoulder, right_hip=right_hip)
    head_pose = third_arg(head=head, left_shoulder=left_shoulder, right_shoulder=right_shoulder)
    

    first = first_coef(angles=angles)
    second = second_coef(btoa=btoa)
    third = third_coef(head_pose=head_pose)

    #print(first, '|', second, '|', third)
    #print(f'Вероятнось того что человек лежит - {(first + second + third) / 3}\n')
    #print(f"Затраченное время:{datetime.datetime.now() - start_time}")
    #print('-' * 50, '\n')

    return (first + second + third) / 3

def first_arg(left_shoulder, left_hip, right_shoulder, right_hip):

    alpha = None
    betta = None

    if ([left_shoulder[0], left_shoulder[1]] != [0.0, 0.0]) and ([left_hip[0], left_hip[1]] != [0.0, 0.0]):
        lyy = abs(left_shoulder[1] - left_hip[1])
        lxx = abs(left_shoulder[0] - left_hip[0])
        lc = math.sqrt(math.pow(lyy, 2) + math.pow(lxx, 2))
        alpha = (int(math.acos(lyy / lc) * 180 / math.pi))

    
    if ([right_shoulder[0], right_shoulder[1]] != [0.0, 0.0]) and ([right_hip[0], right_hip[1]] != [0.0, 0.0]):
        ryy = abs(right_shoulder[1] - right_hip[1])
        rxx = abs(right_shoulder[0] - right_hip[0])
        rc = math.sqrt(math.pow(ryy, 2) + math.pow(rxx, 2))
        betta = (int(math.acos(ryy / rc) * 180 / math.pi))
    
    #print(f'left angle is {alpha}°')
    #print(f'right angle is {betta}°\n')
    return [alpha, betta]

def second_arg(left_shoulder, left_hip, right_shoulder, right_hip):

    c = None
    d = None

    if ([left_shoulder[0], left_shoulder[1]] != [0.0, 0.0]) and ([left_hip[0], left_hip[1]] != [0.0, 0.0]):
        a = abs(left_shoulder[0] - left_hip[0])
        b = abs(left_shoulder[1] - left_hip[1])
        c = b / a
        
    if ([right_shoulder[0], right_shoulder[1]] != [0.0, 0.0]) and ([right_hip[0], right_hip[1]] != [0.0, 0.0]):
        a = abs(right_shoulder[0] - right_hip[0])
        b = abs(right_shoulder[1] - right_hip[1])
        d = b / a
    
    #print(f'left b/a is {c}')
    #print(f'right b/a is {d}\n')

    return [c, d]

def third_arg(head, left_shoulder, right_shoulder):

    is_above = above(head=head, left_shoulder=left_shoulder, right_shoulder=right_shoulder)
    is_right = rigth(head=head, left_shoulder=left_shoulder, right_shoulder=right_shoulder)
    is_left = left(head=head, left_shoulder=left_shoulder, right_shoulder=right_shoulder)

    if (is_above):
        #print("Head is above shoulders\n")
        return False
    elif (not is_above and is_right):
        #print("head is on right\n")
        return True
    elif (not is_above and is_left):
        #print("head is on left\n")
        return True
    elif (not is_above and not is_right and not is_left):
        #print("Head is under shoulders\n")
        return 2

def above(head, left_shoulder, right_shoulder):
    return (head[1] < left_shoulder[1]) and (head[1] < right_shoulder[1])

def rigth(head, left_shoulder, right_shoulder):
    first = ((head[0] >= left_shoulder[0] and head[0] >= right_shoulder[0]) and \
        (right_shoulder[1] <= head[1] and left_shoulder[1] >= head[1]))
    second = ((head[0] >= left_shoulder[0] and head[0] <= right_shoulder[0]) and \
        (right_shoulder[1] <= head[1] and left_shoulder[1] >= head[1]))
    
    return first or second

def left(head, left_shoulder, right_shoulder):
    first = ((head[0] <= left_shoulder[0] and head[0] <= right_shoulder[0]) and \
        (right_shoulder[1] >= head[1] and left_shoulder[1] <= head[1]))
    second = ((head[0] >= left_shoulder[0] and head[0] <= right_shoulder[0]) and \
        (right_shoulder[1] >= head[1] and left_shoulder[1] <= head[1]))
    
    return first or second

def first_coef(angles):
    try: first = angles[0] // 30 > 0 
    except: first = 0.25
    try: second = angles[1] // 30 > 0
    except: second = 0.25

    return (int(first) + int(second)) / 2

def second_coef(btoa):
    try: first = btoa[0] / 1 < 1
    except: first = 0.25
    try: second = btoa[1] / 1 < 1
    except: second = 0.25

    return (int(first) + int(second)) / 2

def third_coef(head_pose):
    return head_pose / 2

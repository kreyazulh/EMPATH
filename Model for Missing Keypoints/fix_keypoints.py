import cv2
import mediapipe as mp
import numpy as np
import sys
from translation import translate
from prev_and_next import generate_new_list
import copy
import time
from guppy import hpy

h = hpy()

# working with google-built structures are a bit complicated. Refer to comments sometimes

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

start_time = time.time()

cap = cv2.VideoCapture(sys.argv[1])

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
res_mult = 2
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate: ")
print(frame_rate)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ")
print(length)


outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_detected_after_edit.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), frame_rate, (frame_width*res_mult, frame_height*res_mult))

max_right = 0
max_left = 0
count_right = 0
count_left = 0
right = 0
left = 0
right_hand_miss = []
left_hand_miss = []
i =0
j =0
video = []
all_image = []
results = None
batch = 2 * frame_rate  #declare desired batch size here
fix = False
next_fix = False
common_elements = []
add = True

def find_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = list(set1.intersection(set2))
    return common_elements


while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.resize(image, (frame_width*res_mult, frame_height*res_mult))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    all_image.append([i, image, results])
    i=i+1
    j=j+1

    if(i >0 and i%batch==0 and results.right_hand_landmarks != None and results.left_hand_landmarks != None and fix==False):
        fix = True
    elif(i >0 and i%batch==0 and (results.right_hand_landmarks == None or results.left_hand_landmarks == None) and fix==False):
        next_fix = True
    elif(next_fix==True) and (results.right_hand_landmarks != None and results.left_hand_landmarks != None):
        fix = True
        next_fix = False
    elif(j==length):
        fix = True
        next_fix = False


    if (results.right_hand_landmarks == None):
        count_right = count_right+1
        if(count_right>max_right):
            max_right = count_right
        right_hand_miss.append(i)
        if(results.left_hand_landmarks != None):
            pose_x = results.pose_landmarks.landmark[15].x
            pose_y = results.pose_landmarks.landmark[15].y
            pose_z = results.pose_landmarks.landmark[15].z
            pose_list = [pose_x, pose_y, pose_z]

            left_hand_x = results.left_hand_landmarks.landmark[0].x
            left_hand_y = results.left_hand_landmarks.landmark[0].y
            left_hand_z = results.left_hand_landmarks.landmark[0].z
            left_hand_list = [left_hand_x, left_hand_y, left_hand_z]

            translate_x = left_hand_x - pose_x
            translate_y = left_hand_y - pose_y
            translate_z = left_hand_z - pose_z
            translate_list = [translate_x, translate_y, translate_z]
            new_pose_list = translate([pose_list], translate_list)

            setattr(results.pose_landmarks.landmark[15], 'x', new_pose_list[0])
            setattr(results.pose_landmarks.landmark[15], 'y', new_pose_list[1])
            setattr(results.pose_landmarks.landmark[15], 'z', new_pose_list[2])


            setattr(results.pose_landmarks.landmark[17], 'x', results.left_hand_landmarks.landmark[17].x)
            setattr(results.pose_landmarks.landmark[17], 'y', results.left_hand_landmarks.landmark[17].y)
            setattr(results.pose_landmarks.landmark[17], 'z', results.left_hand_landmarks.landmark[17].z)

            setattr(results.pose_landmarks.landmark[19], 'x', results.left_hand_landmarks.landmark[5].x)
            setattr(results.pose_landmarks.landmark[19], 'y', results.left_hand_landmarks.landmark[5].y)
            setattr(results.pose_landmarks.landmark[19], 'z', results.left_hand_landmarks.landmark[5].z)

            setattr(results.pose_landmarks.landmark[21], 'x', results.left_hand_landmarks.landmark[9].x)
            setattr(results.pose_landmarks.landmark[21], 'y', results.left_hand_landmarks.landmark[9].y)
            setattr(results.pose_landmarks.landmark[21], 'z', results.left_hand_landmarks.landmark[9].z)

        elif(results.left_hand_landmarks == None):
            left_hand_miss.append(i)

        continue
    else:
        # print(results.right_hand_landmarks.landmark[0])  #to access the first Value
        # results.right_hand_landmarks.landmark[0].setter('x', 1)
        # print(type(results.right_hand_landmarks.landmark[0]))
        # print(getattr(results.right_hand_landmarks.landmark[0], 'x'))
        # setattr(results.right_hand_landmarks.landmark[0], 'x', 0)

        pose_x = results.pose_landmarks.landmark[16].x
        pose_y = results.pose_landmarks.landmark[16].y
        pose_z = results.pose_landmarks.landmark[16].z
        pose_list = [pose_x, pose_y, pose_z]

        right_hand_x = results.right_hand_landmarks.landmark[0].x
        right_hand_y = results.right_hand_landmarks.landmark[0].y
        right_hand_z = results.right_hand_landmarks.landmark[0].z
        right_hand_list = [right_hand_x, right_hand_y, right_hand_z]

        translate_x = right_hand_x - pose_x
        translate_y = right_hand_y - pose_y
        translate_z = right_hand_z - pose_z
        translate_list = [translate_x, translate_y, translate_z]
        new_pose_list = translate([pose_list], translate_list)

        setattr(results.pose_landmarks.landmark[16], 'x', new_pose_list[0])
        setattr(results.pose_landmarks.landmark[16], 'y', new_pose_list[1])
        setattr(results.pose_landmarks.landmark[16], 'z', new_pose_list[2])

        setattr(results.pose_landmarks.landmark[18], 'x', results.right_hand_landmarks.landmark[17].x)
        setattr(results.pose_landmarks.landmark[18], 'y', results.right_hand_landmarks.landmark[17].y)
        setattr(results.pose_landmarks.landmark[18], 'z', results.right_hand_landmarks.landmark[17].z)

        setattr(results.pose_landmarks.landmark[20], 'x', results.right_hand_landmarks.landmark[5].x)
        setattr(results.pose_landmarks.landmark[20], 'y', results.right_hand_landmarks.landmark[5].y)
        setattr(results.pose_landmarks.landmark[20], 'z', results.right_hand_landmarks.landmark[5].z)

        setattr(results.pose_landmarks.landmark[22], 'x', results.right_hand_landmarks.landmark[9].x)
        setattr(results.pose_landmarks.landmark[22], 'y', results.right_hand_landmarks.landmark[9].y)
        setattr(results.pose_landmarks.landmark[22], 'z', results.right_hand_landmarks.landmark[9].z)

    if (results.left_hand_landmarks == None):
        count_left = count_left+1
        if(count_left>max_left):
            max_left = count_left
        left_hand_miss.append(i)
        if(results.right_hand_landmarks != None):
            pose_x = results.pose_landmarks.landmark[16].x
            pose_y = results.pose_landmarks.landmark[16].y
            pose_z = results.pose_landmarks.landmark[16].z
            pose_list = [pose_x, pose_y, pose_z]

            right_hand_x = results.right_hand_landmarks.landmark[0].x
            right_hand_y = results.right_hand_landmarks.landmark[0].y
            right_hand_z = results.right_hand_landmarks.landmark[0].z
            right_hand_list = [right_hand_x, right_hand_y, right_hand_z]

            translate_x = right_hand_x - pose_x
            translate_y = right_hand_y - pose_y
            translate_z = right_hand_z - pose_z
            translate_list = [translate_x, translate_y, translate_z]
            new_pose_list = translate([pose_list], translate_list)

            setattr(results.pose_landmarks.landmark[16], 'x', new_pose_list[0])
            setattr(results.pose_landmarks.landmark[16], 'y', new_pose_list[1])
            setattr(results.pose_landmarks.landmark[16], 'z', new_pose_list[2])

            setattr(results.pose_landmarks.landmark[18], 'x', results.right_hand_landmarks.landmark[17].x)
            setattr(results.pose_landmarks.landmark[18], 'y', results.right_hand_landmarks.landmark[17].y)
            setattr(results.pose_landmarks.landmark[18], 'z', results.right_hand_landmarks.landmark[17].z)

            setattr(results.pose_landmarks.landmark[20], 'x', results.right_hand_landmarks.landmark[5].x)
            setattr(results.pose_landmarks.landmark[20], 'y', results.right_hand_landmarks.landmark[5].y)
            setattr(results.pose_landmarks.landmark[20], 'z', results.right_hand_landmarks.landmark[5].z)

            setattr(results.pose_landmarks.landmark[22], 'x', results.right_hand_landmarks.landmark[9].x)
            setattr(results.pose_landmarks.landmark[22], 'y', results.right_hand_landmarks.landmark[9].y)
            setattr(results.pose_landmarks.landmark[22], 'z', results.right_hand_landmarks.landmark[9].z)

        elif(results.right_hand_landmarks == None):
            right_hand_miss.append(i)
        continue

    else:
        pose_x = results.pose_landmarks.landmark[15].x
        pose_y = results.pose_landmarks.landmark[15].y
        pose_z = results.pose_landmarks.landmark[15].z
        pose_list = [pose_x, pose_y, pose_z]

        left_hand_x = results.left_hand_landmarks.landmark[0].x
        left_hand_y = results.left_hand_landmarks.landmark[0].y
        left_hand_z = results.left_hand_landmarks.landmark[0].z
        left_hand_list = [left_hand_x, left_hand_y, left_hand_z]

        translate_x = left_hand_x - pose_x
        translate_y = left_hand_y - pose_y
        translate_z = left_hand_z - pose_z
        translate_list = [translate_x, translate_y, translate_z]
        new_pose_list = translate([pose_list], translate_list)

        setattr(results.pose_landmarks.landmark[15], 'x', new_pose_list[0])
        setattr(results.pose_landmarks.landmark[15], 'y', new_pose_list[1])
        setattr(results.pose_landmarks.landmark[15], 'z', new_pose_list[2])

        setattr(results.pose_landmarks.landmark[17], 'x', results.left_hand_landmarks.landmark[17].x)
        setattr(results.pose_landmarks.landmark[17], 'y', results.left_hand_landmarks.landmark[17].y)
        setattr(results.pose_landmarks.landmark[17], 'z', results.left_hand_landmarks.landmark[17].z)

        setattr(results.pose_landmarks.landmark[19], 'x', results.left_hand_landmarks.landmark[5].x)
        setattr(results.pose_landmarks.landmark[19], 'y', results.left_hand_landmarks.landmark[5].y)
        setattr(results.pose_landmarks.landmark[19], 'z', results.left_hand_landmarks.landmark[5].z)

        setattr(results.pose_landmarks.landmark[21], 'x', results.left_hand_landmarks.landmark[9].x)
        setattr(results.pose_landmarks.landmark[21], 'y', results.left_hand_landmarks.landmark[9].y)
        setattr(results.pose_landmarks.landmark[21], 'z', results.left_hand_landmarks.landmark[9].z)

    #for face (FACEMESH_TESSELATION, )FACEMESH_CONTOURS)
    #mp_holistic.FACE_CONNECTIONS (which joints connect which)


    if(fix==True):
        common_miss = find_common_elements(right_hand_miss, left_hand_miss)
        #print("Right hand missed frames: ")
        #print(right_hand_miss)
        #right = right + len(right_hand_miss)
        #print("Left hand missed frames: ")
        #print(left_hand_miss)
        #left = left + len(left_hand_miss)
        fix_right_hand_miss = generate_new_list(right_hand_miss)
        # print("Right hand missed frames after fix: ")
        #print(fix_right_hand_miss)
        fix_left_hand_miss = generate_new_list(left_hand_miss)
        # print("Left hand missed frames after fix: ")
        #print(fix_left_hand_miss)

        for j in range(len(fix_right_hand_miss)):
            start = fix_right_hand_miss[j][0]
            end = fix_right_hand_miss[j][1]
            multiplier = 1;
            for k in range(start+1, end):
                all_image[k-1][2].right_hand_landmarks = copy.deepcopy(results.right_hand_landmarks)
                for landmarks in range(len(results.right_hand_landmarks.landmark)):
                    setattr(all_image[k-1][2].right_hand_landmarks.landmark[landmarks], 'x', (all_image[start-1][2].right_hand_landmarks.landmark[landmarks].x+(all_image[end-1][2].right_hand_landmarks.landmark[landmarks].x-all_image[start-1][2].right_hand_landmarks.landmark[landmarks].x)*(multiplier/(end-start))))
                    setattr(all_image[k-1][2].right_hand_landmarks.landmark[landmarks], 'y', (all_image[start-1][2].right_hand_landmarks.landmark[landmarks].y+(all_image[end-1][2].right_hand_landmarks.landmark[landmarks].y-all_image[start-1][2].right_hand_landmarks.landmark[landmarks].y)*(multiplier/(end-start))))
                    setattr(all_image[k-1][2].right_hand_landmarks.landmark[landmarks], 'z', (all_image[start-1][2].right_hand_landmarks.landmark[landmarks].z+(all_image[end-1][2].right_hand_landmarks.landmark[landmarks].z-all_image[start-1][2].right_hand_landmarks.landmark[landmarks].z)*(multiplier/(end-start))))
                multiplier = multiplier+1


                pose_x = all_image[k-1][2].pose_landmarks.landmark[16].x
                pose_y = all_image[k-1][2].pose_landmarks.landmark[16].y
                pose_z = all_image[k-1][2].pose_landmarks.landmark[16].z
                pose_list = [pose_x, pose_y, pose_z]

                right_hand_x = all_image[k-1][2].right_hand_landmarks.landmark[0].x
                right_hand_y = all_image[k-1][2].right_hand_landmarks.landmark[0].y
                right_hand_z = all_image[k-1][2].right_hand_landmarks.landmark[0].z
                right_hand_list = [right_hand_x, right_hand_y, right_hand_z]

                translate_x = right_hand_x - pose_x
                translate_y = right_hand_y - pose_y
                translate_z = right_hand_z - pose_z
                translate_list = [translate_x, translate_y, translate_z]
                new_pose_list = translate([pose_list], translate_list)

                setattr(all_image[k-1][2].pose_landmarks.landmark[16], 'x', new_pose_list[0])
                setattr(all_image[k-1][2].pose_landmarks.landmark[16], 'y', new_pose_list[1])
                setattr(all_image[k-1][2].pose_landmarks.landmark[16], 'z', new_pose_list[2])

                setattr(all_image[k-1][2].pose_landmarks.landmark[18], 'x', all_image[k-1][2].right_hand_landmarks.landmark[17].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[18], 'y', all_image[k-1][2].right_hand_landmarks.landmark[17].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[18], 'z', all_image[k-1][2].right_hand_landmarks.landmark[17].z)

                setattr(all_image[k-1][2].pose_landmarks.landmark[20], 'x', all_image[k-1][2].right_hand_landmarks.landmark[5].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[20], 'y', all_image[k-1][2].right_hand_landmarks.landmark[5].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[20], 'z', all_image[k-1][2].right_hand_landmarks.landmark[5].z)

                setattr(all_image[k-1][2].pose_landmarks.landmark[22], 'x', all_image[k-1][2].right_hand_landmarks.landmark[9].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[22], 'y', all_image[k-1][2].right_hand_landmarks.landmark[9].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[22], 'z', all_image[k-1][2].right_hand_landmarks.landmark[9].z)


                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].face_landmarks,
                                          mp_holistic.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=0),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=0))

                # mp_drawing.draw_landmarks?? to see the function params
                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=4, circle_radius=1))

                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].left_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=4, circle_radius=1))

                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=1))

                #video.append([k - 1, all_image[k - 1][1]])
                for ins in common_miss:
                    if ins == k:
                        add = False
                        break
                if(add==True):
                    video.append([k - 1, all_image[k - 1][1]])
                add = True

            right_hand_miss = []



        for j in range(len(fix_left_hand_miss)):
            start = fix_left_hand_miss[j][0]
            end = fix_left_hand_miss[j][1]
            multiplier = 1;
            for k in range(start+1, end):
                all_image[k-1][2].left_hand_landmarks = copy.deepcopy(results.left_hand_landmarks)
                for landmarks in range(len(results.left_hand_landmarks.landmark)):
                    setattr(all_image[k-1][2].left_hand_landmarks.landmark[landmarks], 'x', (all_image[start-1][2].left_hand_landmarks.landmark[landmarks].x+(all_image[end-1][2].left_hand_landmarks.landmark[landmarks].x-all_image[start-1][2].left_hand_landmarks.landmark[landmarks].x)*(multiplier/(end-start))))
                    setattr(all_image[k-1][2].left_hand_landmarks.landmark[landmarks], 'y', (all_image[start-1][2].left_hand_landmarks.landmark[landmarks].y+(all_image[end-1][2].left_hand_landmarks.landmark[landmarks].y-all_image[start-1][2].left_hand_landmarks.landmark[landmarks].y)*(multiplier/(end-start))))
                    setattr(all_image[k-1][2].left_hand_landmarks.landmark[landmarks], 'z', (all_image[start-1][2].left_hand_landmarks.landmark[landmarks].z+(all_image[end-1][2].left_hand_landmarks.landmark[landmarks].z-all_image[start-1][2].left_hand_landmarks.landmark[landmarks].z)*(multiplier/(end-start))))
                multiplier = multiplier+1

                pose_x = all_image[k-1][2].pose_landmarks.landmark[15].x
                pose_y = all_image[k-1][2].pose_landmarks.landmark[15].y
                pose_z = all_image[k-1][2].pose_landmarks.landmark[15].z
                pose_list = [pose_x, pose_y, pose_z]

                left_hand_x = all_image[k-1][2].left_hand_landmarks.landmark[0].x
                left_hand_y = all_image[k-1][2].left_hand_landmarks.landmark[0].y
                left_hand_z = all_image[k-1][2].left_hand_landmarks.landmark[0].z
                left_hand_list = [left_hand_x, left_hand_y, left_hand_z]

                translate_x = left_hand_x - pose_x
                translate_y = left_hand_y - pose_y
                translate_z = left_hand_z - pose_z
                translate_list = [translate_x, translate_y, translate_z]
                new_pose_list = translate([pose_list], translate_list)

                setattr(all_image[k-1][2].pose_landmarks.landmark[15], 'x', new_pose_list[0])
                setattr(all_image[k-1][2].pose_landmarks.landmark[15], 'y', new_pose_list[1])
                setattr(all_image[k-1][2].pose_landmarks.landmark[15], 'z', new_pose_list[2])

                setattr(all_image[k-1][2].pose_landmarks.landmark[17], 'x', all_image[k-1][2].left_hand_landmarks.landmark[17].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[17], 'y', all_image[k-1][2].left_hand_landmarks.landmark[17].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[17], 'z', all_image[k-1][2].left_hand_landmarks.landmark[17].z)

                setattr(all_image[k-1][2].pose_landmarks.landmark[19], 'x', all_image[k-1][2].left_hand_landmarks.landmark[5].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[19], 'y', all_image[k-1][2].left_hand_landmarks.landmark[5].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[19], 'z', all_image[k-1][2].left_hand_landmarks.landmark[5].z)

                setattr(all_image[k-1][2].pose_landmarks.landmark[21], 'x', all_image[k-1][2].left_hand_landmarks.landmark[9].x)
                setattr(all_image[k-1][2].pose_landmarks.landmark[21], 'y', all_image[k-1][2].left_hand_landmarks.landmark[9].y)
                setattr(all_image[k-1][2].pose_landmarks.landmark[21], 'z', all_image[k-1][2].left_hand_landmarks.landmark[9].z)


                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].face_landmarks,
                                          mp_holistic.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=0),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=0))

                # mp_drawing.draw_landmarks?? to see the function params
                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=4, circle_radius=1))

                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].left_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=4, circle_radius=1))

                mp_drawing.draw_landmarks(all_image[k-1][1], all_image[k-1][2].pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=1))

                video.append([k-1, all_image[k-1][1]])

        left_hand_miss = []
        #print(h.heap())   #uncomment this to see heap allocation in memory

        for data in range(len(all_image)-1):  # Sort based on frame count before writing
            all_image[data][1] = None
            all_image[data][2]=None

        for frame_data in sorted(video, key=lambda x: x[0]):  # Sort based on frame count before writing
            out.write(frame_data[1])
        video = []
        fix = False


    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=0), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=0))

    #mp_drawing.draw_landmarks?? to see the function params
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=4, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=4, circle_radius=1))

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=4, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=4, circle_radius=1))

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=1))

    video.append([i, image])


for frame_data in sorted(video, key=lambda x: x[0]):  # Sort based on frame count before writing
    out.write(frame_data[1])
    #print(frame_data[0])

#print("Common missed frames: ")
#print(common_miss)

# print("Max right: ")
# print(max_right)
# print("Max left: ")
# print(max_left)
# print("total misses")
# print(right+left)
# print(left)
# print(right)
#print("total miss")
#print(count_right+count_left)
holistic.close()
cap.release()
out.release()


print("--- %s seconds ---" % (time.time() - start_time))
# print(h.heap())

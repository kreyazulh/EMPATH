import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from guppy import hpy

h = hpy()


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#you can change these parameters
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

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind(
    '/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_before.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), frame_rate, (frame_width*res_mult, frame_height*res_mult))

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.resize(image, (frame_width*2, frame_height*2))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)


    #for face (FACEMESH_TESSELATION, )FACEMESH_CONTOURS)
    #mp_holistic.FACE_CONNECTIONS (which joints connect which)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
    # print(h.heap())
    out.write(image)

holistic.close()
cap.release()
out.release()

print("--- %s seconds ---" % (time.time() - start_time))

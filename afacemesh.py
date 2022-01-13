# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:41:22 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:37:28 2021

@author: 
detail:
    这个文件是用来验证算法的，而且可以打印显示关键点的位置等，如果需要实时找关键点可以用这个 只需要改164行的代码就行了
    主要是用python取做算法的精度等验证比较快 再把算法转C++等就不会因为没有验证导致浪费时间
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

landmark_pre_list = []
MY_FACE_CONNECTIONS = np.array([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    # (10, 338),
    # (338, 297),
    # (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    # (67, 109),
    # (109, 10)
])[:,0]

easy_connections = [
    ( 0 , 1 ),
    ( 1 , 2 ),
    ( 2 , 3 ),
    ( 3 , 4 ),
    ( 4 , 5 ),
    ( 5 , 6 ),
    ( 6 , 7 ),
    ( 7 , 8 ),
    ( 8 , 9 ),
    (9,19),
    ( 10 , 11 ),
    ( 11 , 12 ),
    ( 12 , 13 ),
    ( 13 , 14 ),
    ( 14 , 15 ),
    ( 15 , 16 ),
    ( 16 , 17 ),
    ( 17 , 18 ),
    ( 18 , 19 ),

    ( 20 , 21 ),
    ( 21 , 22 ),
    ( 22 , 23 ),
    ( 23 , 24 ),
    ( 24 , 25 ),
    ( 25 , 26 ),
    ( 26 , 27 ),
    ( 27 , 28 ),
    ( 28 , 29 ),
    (29,39),
    ( 30 , 31 ),
    ( 31 , 32 ),
    ( 32 , 33 ),
    ( 33 , 34 ),
    ( 34 , 35 ),
    ( 35 , 36 ),
    ( 36 , 37 ),
    ( 37 , 38 ),
    ( 38 , 39 ),

    ( 40 , 41 ),
    ( 41 , 42 ),
    ( 42 , 43 ),
    ( 43 , 44 ),
    ( 44 , 45 ),
    ( 45 , 46 ),
    ( 46 , 47 ),
    
    ( 48 , 49 ),
    ( 49 , 50 ),
    ( 50 , 51 ),
    ( 51 , 52 ),
    ( 52 , 53 ),
    ( 53 , 54 ),
    ( 54 , 55 ),
    
    ( 56 , 57 ),
    ( 57 , 58 ),
    ( 58 , 59 ),
    ( 60 , 61 ),
    ( 61 , 62 ),
    ( 62 , 63 ),
    
    ( 64 , 65 ),
    ( 65 , 66 ),
    ( 66 , 67 ),
    ( 67 , 68 ),
    ( 68 , 69 ),
    ( 69 , 70 ),
    ( 70 , 71 ),
   
    ( 72 , 73 ),
    ( 73 , 74 ),
    ( 74 , 75 ),
    ( 75 , 76 ),
    ( 76 , 77 ),
    ( 77 , 78 ),
    ( 78 , 79 ),

    (80,81),
    ( 81 , 82 ),
    ( 82 , 83 ),
    ( 84 , 85 ),
    ( 85 , 86 ),
    ( 86 , 87 ),

    (88,89),
    (89,90),
    (90,91),
    ( 91 , 92 ),
    ( 92 , 93 ),
    ( 93 , 94 ),
    ( 94 , 95 ),
    ( 95 , 96 ),
    ( 96 , 97 ),
    ( 97 , 98 ),
    ( 98 , 99 ),
    ( 99 , 100 ),
    ( 100 , 101 ),
    ( 101 , 102 ),
    ( 102 , 103 ),
    ( 103 , 104 ),
    ( 104 , 105 ),
    ( 105 , 106 ),
    ( 106 , 107 ),
    ( 107 , 108 ),
    ( 108 , 109 ),
    ( 109 , 110 ),
    ( 110 , 111 ),
    ( 111 , 112 ),
    ( 112 , 113 ),
    ( 113 , 114 ),
    ( 114 , 115 ),
    ( 115 , 116 ),
    ( 116 , 117 ),
    ( 117 , 118 ),
    ( 118 , 88 ),
    # ( 119 , 120 ),
    # ( 120 , 121 ),
    
   # ( 118 , 119 ),
]
# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./video/face.avi")
w_scal = 1
h_scal = 1
index = 0
sign = 0
show_flag = False
MAX_NUM_FACES = 2
landmark_pre_list = [[] for i in range(MAX_NUM_FACES)]

with mp_face_mesh.FaceMesh(max_num_faces = 2,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(640,480))
        
        image.flags.writeable = False
        
        results = face_mesh.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = image.copy()
        image_kalman = image.copy()
        image_linefitting = image.copy()
        #image3 = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
        
        #hist = np.array(image3)
        if results.multi_face_landmarks:

            

            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                

                temp = []
                for face_index in MY_FACE_CONNECTIONS:
                    temp.append(face_landmarks.landmark[face_index])
                    
                
                
                image_height, image_width, _ = image.shape          
                if len(landmark_pre_list[idx]) != 0 :
                    # 插值拟合降噪
                    line_fitting_result = smooth_point( temp, landmark_pre_list[idx], image_width, image_height)
                    # 光流法降噪
                    landmark_pre_list[idx] = smooth_point_by_caloptflow( temp, landmark_pre_list[idx], image_width, image_height, "facemesh", image)
                    #卡尔曼滤波降噪
                    kalman_result = smooth_point_by_kalman( temp, landmark_pre_list[idx], image_width, image_height, "facemesh", image)
                    
                    save_ans_to_file(temp, "./save/origin_result_face.txt",1)
                    save_ans_to_file(line_fitting_result, "./save/lin_fitting_result_face.txt",1)
                    save_ans_to_file( landmark_pre_list[idx], "./save/optflow_result_face.txt",1)
                    save_ans_to_file(kalman_result, "./save/kalman_result_face.txt",1)

                    my_draw_landmarks(image2,landmark_pre_list[idx],easy_connections, name = "optflow")# mp_face_mesh.FACE_CONNECTIONS)
                    my_draw_landmarks(image, temp, easy_connections, name = "origin")# mp_face_mesh.FACE_CONNECTIONS)
                    my_draw_landmarks(image_kalman, kalman_result, easy_connections, name = "kalman")
                    my_draw_landmarks(image_linefitting, line_fitting_result, easy_connections, name = "line fitting")
                else:
                    landmark_pre_list[idx] = temp
            
            image3 = cv2.resize(image.copy(),(640,480))
            image4 = cv2.resize(image2.copy(),(640,480))
            image_linefitting_resize = cv2.resize(image_linefitting.copy(),(640,480))
            image_kalman_resize = cv2.resize(image_kalman.copy(),(640,480))


            htitch2 = np.hstack((image3, image_linefitting_resize))  
            htitch3 = np.hstack((image4, image_kalman_resize ))
            
            vtitich2 = np.vstack((htitch2, htitch3))

            cv2.imshow('mediapipe orgin 640*480',vtitich2)
            
            show_flag = True
        else:
            show_flag = False
        if(not show_flag):    
            htitch = np.hstack((image, image))  
            vtitich = np.vstack( (htitch, htitch))
            cv2.imshow('mediapipe orgin 640*480',vtitich)
            show_flag = False
          
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
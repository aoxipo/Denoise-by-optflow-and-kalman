# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 08:16:15 2021

@author: Administrator
"""

import cv2
import mediapipe as mp
from utils import *
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

show_flag = False
MAX_NUM_HANDS = 2
landmark_pre_list = [[] for i in range(MAX_NUM_HANDS)]
index = 0
cap = cv2.VideoCapture("./video/hand.avi")
with mp_hands.Hands(
    max_num_hands = MAX_NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image,(640,480))

    image.flags.writeable = False

    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = image.copy()
    image_kalman = image.copy()
    image_linefitting = image.copy()
    #image3 = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)

    #hist = np.array(image3)
    if results.multi_hand_landmarks:
      for idx, hand_landmarks in enumerate( results.multi_hand_landmarks):
        image_height, image_width, _ = image.shape 
        temp = hand_landmarks.landmark  
        if len(landmark_pre_list[idx]) != 0 and index >= 2:

          landmark_pre_list[idx] = smooth_point( temp, landmark_pre_list[idx], image_width, image_height)
          
          # 插值拟合降噪
          line_fitting_result = smooth_point( temp, landmark_pre_list[idx], image_width, image_height)
          # 光流法降噪
          landmark_pre_list[idx] = smooth_point_by_caloptflow( temp, landmark_pre_list[idx], image_width, image_height, "handpose", image)
          #卡尔曼滤波降噪
          kalman_result = smooth_point_by_kalman( temp, landmark_pre_list[idx], image_width, image_height, "handpose", image)

          save_ans_to_file(temp, "./save/origin_result_hand.txt",1)
          save_ans_to_file(line_fitting_result, "./save/lin_fitting_result_hand.txt",1)
          save_ans_to_file(landmark_pre_list[idx], "./save/optflow_result_hand.txt",1)
          save_ans_to_file(kalman_result, "./save/kalman_result_hand.txt",1)

          #print(len(landmark_pre_list[idx]))
          my_draw_landmarks(image, temp ,mp_hands.HAND_CONNECTIONS,landmark_circle_radius = 4,spec_thickness = 2,need_cal_eular = False,landmark_thickness=-1, name = "origin" )
          my_draw_landmarks(image2,landmark_pre_list[idx],mp_hands.HAND_CONNECTIONS, landmark_circle_radius = 4,spec_thickness = 2,need_cal_eular = False,landmark_thickness=-1, name = "optflow")
          my_draw_landmarks(image_kalman, kalman_result, mp_hands.HAND_CONNECTIONS,landmark_circle_radius = 4,spec_thickness = 2,need_cal_eular = False,landmark_thickness=-1, name = "kalman")
          my_draw_landmarks(image_linefitting, line_fitting_result, mp_hands.HAND_CONNECTIONS,landmark_circle_radius = 4,spec_thickness = 2,need_cal_eular = False,landmark_thickness=-1, name = "line fitting")
        else:
          landmark_pre_list[idx] = hand_landmarks.landmark
        index += 1
    
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
    
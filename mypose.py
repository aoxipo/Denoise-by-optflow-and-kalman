# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 08:14:49 2021

@author: Administrator
"""

import cv2
import mediapipe as mp
from utils import pose, my_draw_landmarks, sigmod, smooth_box, smooth, smooth_point, translate_point
import numpy as np
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

show_flag = False
MAX_NUM_POSE = 2
landmark_pre_list = [[] for i in range(MAX_NUM_POSE)]


cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = image.copy()

    if results.pose_landmarks:
      pose_landmarks = results.pose_landmarks
      image_height, image_width, _ = image.shape   
      idx = 0
      if len(landmark_pre_list[idx]) != 0:
        landmark_pre_list[idx] = smooth_point( pose_landmarks.landmark, landmark_pre_list[idx], image_width, image_height)
        mp_drawing.draw_landmarks(image,pose_landmarks,mp_pose.POSE_CONNECTIONS)
        my_draw_landmarks(image2,landmark_pre_list[idx],mp_pose.POSE_CONNECTIONS, landmark_circle_radius = 4,spec_thickness = 2,need_cal_eular = False,landmark_thickness=-1)
      else:
        landmark_pre_list[idx] = pose_landmarks.landmark
        
      image3 = cv2.resize(image.copy(),(640,480))
      image4 = cv2.resize(image2.copy(),(640,480))
      htitch2= np.hstack((image3, image4))  
      cv2.imshow('mediapipe orgin 640*480',htitch2)
      show_flag = True
    else:
      show_flag = False
    if(not show_flag):    
        htitch = np.hstack((image, image))  
        cv2.imshow('mediapipe orgin 640*480',htitch)
        show_flag = False
          
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
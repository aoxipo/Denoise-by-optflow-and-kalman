# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:54:28 2021

@author: Administrator
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
mp_pose = mp.solutions.pose
 
 
# # For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      cv2.destroyAllWindows()
      cap.release()
      break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(results.pose_landmarks)
    mp_drawing.draw_landmarks(  image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == 27:
      cv2.destroyAllWindows()
      cap.release()
      break

landmark_pre_list = None
cap = cv2.VideoCapture("./video/hand.avi")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image,(640,480))
    
    image.flags.writeable = False
    
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = image.copy()
    image_kalman = image.copy()
    image_linefitting = image.copy()
    if results.pose_landmarks:
      image_height, image_width, _ = image.shape          
      if landmark_pre_list is not None :
        # 插值拟合降噪
        temp = results.pose_landmarks
        line_fitting_result = smooth_point( temp, landmark_pre_list, image_width, image_height)
        # 光流法降噪
        landmark_pre_list = smooth_point_by_caloptflow( temp, landmark_pre_list, image_width, image_height, "facemesh", image)
        #卡尔曼滤波降噪
        kalman_result = smooth_point_by_kalman( temp, landmark_pre_list, image_width, image_height, "facemesh", image)
        
        save_ans_to_file(temp, "./save/origin_result_hand.txt",1)
        save_ans_to_file(line_fitting_result, "./save/lin_fitting_result_hand.txt",1)
        save_ans_to_file( landmark_pre_list, "./save/optflow_result_hand.txt",1)
        save_ans_to_file(kalman_result, "./save/kalman_result_hand.txt",1)

        my_draw_landmarks(image2,landmark_pre_list,mp_pose.POSE_CONNECTIONS, name = "optflow")# mp_face_mesh.FACE_CONNECTIONS)
        my_draw_landmarks(image, temp, mp_pose.POSE_CONNECTIONS, name = "origin")# mp_face_mesh.FACE_CONNECTIONS)
        my_draw_landmarks(image_kalman, kalman_result, mp_pose.POSE_CONNECTIONS, name = "kalman")
        my_draw_landmarks(image_linefitting, line_fitting_result, mp_pose.POSE_CONNECTIONS, name = "line fitting")
      else:
        landmark_pre_list = results.pose_landmarks
  
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
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
from utils import pose, my_draw_landmarks, sigmod, smooth_box, smooth, smooth_point, translate_point

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

landmark_pre_list = []


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
fig = plt.figure(figsize=(16,12))
ax = fig.gca(projection="3d")
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
          # If loading a video, use 'break' instead of 'continue'.
            continue
    
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(640,480))
        width  = image.shape[1]
        height = image.shape[0]
        
        x = np.arange(0,image.shape[1],1)
        y = np.arange(0,image.shape[0],1)
        x, y = np.meshgrid(x, y)
        
       
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        
        results = face_mesh.process(image)
    
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = image.copy()
        image3 = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
        
        hist = np.array(image3)
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                #print(idx)
                image_height, image_width, _ = image.shape          
                if len(landmark_pre_list[idx]) != 0 :
                    landmark_pre_list[idx] = smooth_point( face_landmarks.landmark, landmark_pre_list[idx], image_width, image_height)
                    my_draw_landmarks(image2,landmark_pre_list[idx],mp_face_mesh.FACE_CONNECTIONS)
                    my_draw_landmarks(image, face_landmarks.landmark, mp_face_mesh.FACE_CONNECTIONS)
                    #print(landmark_pre_list[idx][5],landmark_pre_list[idx][10])
                    #for landmark in landmark_pre_list[idx]:
                    #    hist[int(landmark.x * width)][int(landmark.y * height)] = hist[int(landmark.x * width)][int(landmark.y * height)] - int(landmark.z * width)
                    #surf = ax.plot_surface(x, y, hist, cmap=cm.coolwarm)
                    #ax.zaxis.set_major_locator(LinearLocator(10))  # z轴网格线的疏密，刻度的疏密，20表示刻度的个数
                    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 将z的value字符串转为float，保留2位小数
                    #ax.set_xlabel('x', size=15)
                    #ax.set_ylabel('y', size=15)
                    #ax.set_zlabel('z', size=15)
                    
                    #ax.set_title("Surface plot", weight='bold', size=20)
                    
                    # 添加右侧的色卡条
                    #fig.colorbar(surf, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
                    #plt.show()
                    #cv2.putText(image, str(index), (int( face_landmarks.landmark[index].x*image_width), int(face_landmarks.landmark[index].y*image_height) ) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                    #index = (index + 1)%468
                    #print(index)
                    
                else:
                    landmark_pre_list[idx] = face_landmarks.landmark
            
           
            
            #htitch= np.hstack((image, image2))  
            #cv2.imshow('mediapipe orgin small 640*480',htitch)        
            image3 = cv2.resize(image.copy(),(1280,960))
            image4 = cv2.resize(image2.copy(),(1280,960))
            htitch2= np.hstack((image3, image4))  
            cv2.imshow('mediapipe orgin 1280*960',htitch2)
            show_flag = True
            
            '''mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                '''
            #cv2.imshow('MediaPipe FaceMesh1', image)
            #image = cv2.resize(image,(1600,1400))
            #cv2.imshow('MediaPipe FaceMesh', image)
        else:
            show_flag = False
        if(not show_flag):    
            htitch = np.hstack((image, image))  
            cv2.imshow('mediapipe orgin small 640*480',htitch)
            #image = cv2.resize(image,(1280,960))
            #htitch2 = np.hstack((image, image))  
            #cv2.imshow('mediapipe orgin 1280*960',htitch2)
            show_flag = False
          
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
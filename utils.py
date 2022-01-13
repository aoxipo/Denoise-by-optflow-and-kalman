# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:14:20 2021

@author: Administrator
"""
import cv2
import numpy as np
import math
import copy

def move_judge():
  def deco(func):
    def _dec(*args, **kwargs):
      
      landmark = args[0]
      landmark_pre = args[1]
      width = args[2] # 640
      height = args[3] # 480
      if(func.__name__  == "smooth_point_by_caloptflow"):
        name = args[4]
        frame = args[5]
      
      variance = 0
      mean = 0
      diffs = []

      rect_landmark = []
      for index in range(len(landmark)):
        diff = math.sqrt((landmark[index].x*width  - landmark_pre[index].x*width )**2 +  (landmark[index].y*height - landmark_pre[index].y *height )**2)
        rect_landmark.append( [landmark[index].x * width, landmark[index].y * height]  )
        mean += diff
        diffs.append(diff)
      mean = mean / len(landmark)
      for diff in diffs:
        variance += (diff - mean)**2
      variance /= (25 * len(landmark))
      
      _,_, rect_w, rect_h = cv2.boundingRect(np.array( rect_landmark, dtype = np.float32 ))

      diffdlib = rect_h*rect_w/(width*height)

      #print("{:.9f}, {:.9f}".format(variance, diffdlib))
      
      if( variance < diffdlib*1.5):
        result = func(*args, **kwargs)
      else:
        if(func.__name__  == "smooth_point_by_caloptflow"):
          filiter = optflow(name)
          filiter.set_old_frame(frame)

        result = landmark
      return result
    return _dec
  return deco

class pose:
    def __init__(self):
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                             [1.330353, 7.122144, 6.903745],
                             [-1.330353, 7.122144, 6.903745],
                             [-6.825897, 6.760612, 4.402142],
                             [5.311432, 5.485328, 3.987654],
                             [1.789930, 5.393625, 4.413414],
                             [-1.789930, 5.393625, 4.413414],
                             [-5.311432, 5.485328, 3.987654],
                             [2.005628, 1.409845, 6.165652],
                             [-2.005628, 1.409845, 6.165652]])
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])
        
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
    def get_head_pose(self, shape, img):
        h,w,_=img.shape
        K = [w, 0.0, w//2,
             0.0, w, h//2,
             0.0, 0.0, 1.0]
        # Assuming no lens distortion
        D = [0, 0, 0.0, 0.0, 0]
    
        cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    
    
        image_pts = np.float32([shape[46], shape[55], shape[285], shape[276], shape[33],
                                shape[173], shape[398], shape[263], shape[102], shape[331]])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, cam_matrix, dist_coeffs)
        #print(cam_matrix)
        #print(dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)
    
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    
        return reprojectdst, euler_angle
    
    def draw_pic(self, landmarks, img_show):
        reprojectdst, euler_angle=self.get_head_pose(landmarks, img_show)
        #for start, end in self.line_pairs:
        #    cv2.line(img_show, reprojectdst[start], reprojectdst[end], (0, 0, 255),2)
        x = 0
        y = 0
        for land in reprojectdst:
            x = x + land[0]
            y = y + land[1]
        x = x/8
        y = y/8
        #print("x:",x,"y:",y)
        cv2.circle(img_show, (int(x),int(y)), 1, (255,0,0), 1)
        #print(euler_angle)
        #print("P",euler_angle[0,0])
        #print("y",euler_angle[1,0])
        #print("r",euler_angle[2,0])
        cv2.putText(img_show, "Pitch: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        cv2.putText(img_show, "Yaw: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        cv2.putText(img_show, "Roll: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        return img_show
    
def my_draw_landmarks(
    image: np.ndarray,
    landmark_list: list,
    connections = None,
    landmark_circle_radius = 1 , landmark_color = (0,0,255), landmark_thickness = 1,
    spec_color = (0,255,0),spec_thickness = 1, need_cal_eular = False, name = None):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color, line thickness, and circle radius.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != 3:
    raise ValueError('Input image must contain three channel rgb data.')
   
  image_height, image_width, _ = image.shape
  
  idx_to_coordinates = {}
  landmarks = []
  #print( len(landmark_list))
  for idx, landmark in enumerate(landmark_list):
    
    landmark_px = translate_point(landmark, image_width, image_height)
    landmarks.append(landmark_px)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
      
  if connections is None:
    num_landmarks = len(landmark_list)
    for i in range(num_landmarks-1):
      cv2.line(image, idx_to_coordinates[i],
                 idx_to_coordinates[i+1], spec_color,
                 spec_thickness)
  else:
    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], spec_color,
                 spec_thickness)
      
  
    
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  # min_ = 1
  for key, landmark_px in idx_to_coordinates.items():
    cv2.circle(image, landmark_px, landmark_circle_radius,
               landmark_color, landmark_thickness)
    # if((landmark_list[min_].x -landmark_list[10].x )**2+ (landmark_list[min_].z -landmark_list[10].z)**2 > (landmark_list[key].x -landmark_list[10].x)**2 + (landmark_list[key].z -landmark_list[10].z)**2 and key != 10):
    #    min_ = key
    # if(key == 6 or key == 419 or key == 196 ):
    #cv2.putText(image, str(key), landmark_px, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255))
    #int(landmark_list[key].z*image_width)
  #landmark_px = (int((landmark_list[5].x + landmark_list[5].x)*image_width/2), int((landmark_list[5].y + landmark_list[5].y)*image_height/2))  
  #cv2.putText(image, str(landmark_px), landmark_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
  if(name is not None):
    cv2.putText(image, name , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
  if (need_cal_eular):
    pose_eular = pose()
    pose_eular.draw_pic(landmarks, image)
  
def translate_point(landmark, image_width, image_height):
    coorx = min(math.floor(landmark.x * image_width), image_width - 1)
    coory = min(math.floor(landmark.y * image_height), image_height - 1)
    return coorx,coory
    
def sigmod( x: float):
    if x < 0:
        x = -x
    x -= 6
    return  1/(1 + np.exp(-x))

def smooth_box(x: float, x1: float,  y: float, y1: float, z: int, y_scale: float,x_scale: float):
    sub = (x - x1)*x_scale
    sub2 = (y - y1)*y_scale
    #print(sub,sub2)
    
    ans_x = x1
    ans_y = y1
    
    if (x1 <= x+3*x_scale and x1 >= x-3*x_scale):    
        ans_x = x
    if (y1 <= y+3*y_scale and y1 >= y-3*y_scale):
        ans_y = y
    
    
    return ans_x/640,ans_y/480,z

def smooth( x: float, x1: float, w1: int = 3, w2: int = 4, scale: int = 1):
    sub = (x - x1)*scale
    #print(sub)
    subs = sigmod(sub)
    xx = x + (sub *subs *1.2 + 0.65)/scale
    xx = (xx*w2 + w1*x1)/( w1+ w2)
    return xx

@move_judge()
def smooth_point( face_landmark: list,  face_landmark_pre: list, width: int, height: int):
    
    x_scale = width/640
    y_scale = height/480

    face_landmark_pre
    face_landmark_backup = copy.deepcopy( face_landmark_pre )

    for i, landmark in enumerate(face_landmark):
        x = smooth(landmark.x * width, face_landmark_pre[i].x * width, scale = x_scale)/width
        y = smooth(landmark.y * height, face_landmark_pre[i].y * height, scale = y_scale)/height
        z = smooth(landmark.z, face_landmark_pre[i].z)
        
        face_landmark_backup[i].x, face_landmark_backup[i].y, face_landmark_backup[i].z = smooth_box(
            face_landmark_pre[i].x * width, x * width, 
             face_landmark_pre[i].y * height, y * height, 
            z,
            x_scale = x_scale,y_scale = y_scale, 
            
            )
         
    return face_landmark_backup



class Kalman(object):
  pool = dict()
  def __new__(cls, data_type, *args,**kwargs):
      
      obj = cls.pool.get(data_type,None)
      if not obj:
          obj = super().__new__(cls,*args, **kwargs)
          cls.pool[data_type] = obj
          obj.tree_type = data_type
          obj.set__init()
      else:
        pass
          #print("find object")
      return obj

  def set__init(self):
      '''
      kalman parameters
      '''
      self.Q = 0.018    #噪声的协方差，也可以理解为两个时刻之间噪声的偏差
      self.R = 0.1        #状态的协方差，可以理解为两个时刻之间状态的偏差
      self.P_k_k1 = 1     #上一时刻状态协方差
      self.Kg = 0         #卡尔曼增益
      self.P_k1_k1 = 1
      self.x_k_k1 = 0     #上一时刻状态值
      self.ADC_OLD_Value = 0    #上一次的ADC值
      self.kalman_adc_old = 0   #上一次的卡尔曼滤波的到的最优估计值

  def process(self, ADC_Value):
      Z_k = ADC_Value          #测量值

      if (abs(self.kalman_adc_old-ADC_Value)>=80):      #上一状态值与此刻测量值差距过大，进行简单的一阶滤波，0618黄金比例可以随意定哦
          x_k1_k1= ADC_Value*0.382 + self.kalman_adc_old*0.618
      else:                 #差距不大直接使用
          x_k1_k1 = self.kalman_adc_old;

      self.x_k_k1 = x_k1_k1      #测量值
      self.P_k_k1 = self.P_k1_k1 + self.Q      #公式二
      self.Kg = self.P_k_k1/(self.P_k_k1 + self.R)  #公式三

      kalman_adc = self.x_k_k1 + self.Kg * (Z_k - self.kalman_adc_old)    #计算最优估计值
      self.P_k1_k1 = (1 - self.Kg)*self.P_k_k1  #公式五
      self.P_k_k1 = self.P_k1_k1     #更新状态协方差

      self.ADC_OLD_Value = ADC_Value   
      self.kalman_adc_old = kalman_adc
      return kalman_adc

class optflow():
  pool = dict()
  def __new__(cls, data_type, *args,**kwargs):
      
      obj = cls.pool.get(data_type,None)
      if not obj:
          obj = super().__new__(cls,*args, **kwargs)
          cls.pool[data_type] = obj
          obj.tree_type = data_type
          obj.set__init()
      else:
        pass
          #print("find object")
      return obj

  def set__init(self):
    self.old_gray = None 
    self.lk_params = dict( winSize  = (25,25),
              maxLevel = 3,
              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
              )

  def low_pass(self, landmark_new, landmark_old):
    dx = landmark_new[0] - landmark_old[0]
    dy = landmark_new[1] - landmark_old[1]
    if( dx*dx + dy*dy > 0.5):
      return 1
    else:
      return 0

  def set_old_frame(self,frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    self.old_gray = frame_gray

  def predict(self, frame, landmark:list, old_landmark:list, name = "test"):
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h,w = frame_gray.shape

    now_landmark = []
    pre_landmark = []

    for i, landmark_ in enumerate(old_landmark):
      # p0.append([[landmark_[0][0], landmark_[0][1]]])
      old_point = copy.deepcopy( [[ landmark_.x*w, landmark_.y*h ]] )
      new_point = copy.deepcopy( [[ landmark[i].x*w, landmark[i].y*h ]] )
      pre_landmark.append(old_point)
      now_landmark.append(new_point)

    pre_landmark = np.array(pre_landmark, dtype = np.float32)
    now_landmark = np.array(now_landmark, dtype = np.float32)

    if(self.old_gray is None):
        self.old_gray = frame_gray
        print("ignore first frame")
        return landmark, 1

        #p0 = np.array(p0).reshape(-1,1,2)
    now_optflow_landmark, st, err = cv2.calcOpticalFlowPyrLK( self.old_gray, frame_gray, pre_landmark, now_landmark, **self.lk_params)

    # for i, landmark_ in enumerate(landmark):
    #   if(st[i] == 1):
    #     landmark_.x = now_optflow_landmark[i][0]
    #     landmark_.y = now_optflow_landmark[i][1]

    self.old_gray = frame_gray.copy()

    now_optflow_landmark = np.squeeze(now_optflow_landmark)
    pre_landmark = np.squeeze(pre_landmark)
    
    #cv2.boundingRect

    for index in range(len(now_optflow_landmark)):
      if(not self.low_pass(now_optflow_landmark[index], pre_landmark[index])):
        now_optflow_landmark[index][0] = pre_landmark[index][0] 
        now_optflow_landmark[index][1] = pre_landmark[index][1] 

    return now_optflow_landmark, 0

@move_judge()
def smooth_point_by_kalman( face_landmark: list,  face_landmark_pre: list, width: int, height: int, name:str, frame):

    kalman_pool = []

    point_number = len(face_landmark)

    face_landmark_backup = copy.deepcopy( face_landmark )

    for i in range(point_number):
      x_filiter = Kalman(name+"_x_"+str(i))
      y_filiter = Kalman(name+"_y_"+str(i))
      kalman_pool.append([x_filiter, y_filiter])

    for i, landmark in enumerate(face_landmark_backup):
        face_landmark_pre[i].x = kalman_pool[i][0].process(landmark.x)
        face_landmark_pre[i].y = kalman_pool[i][1].process(landmark.y)
        face_landmark_pre[i].z = landmark.z

    return face_landmark_backup

@move_judge()
def smooth_point_by_caloptflow( face_landmark: list,  face_landmark_pre: list, width: int, height: int, name:str, frame):
    
    filiter = optflow(name)

    x_scale = width/640
    y_scale = height/480

    filiter_list, is_first_frame = filiter.predict(frame, face_landmark, face_landmark_pre, name)
    

    face_landmark_backup = copy.deepcopy( face_landmark )

    for index, face_landmark_object in enumerate(face_landmark_backup):
      if(is_first_frame):
        face_landmark_object.x = filiter_list[index].x
        face_landmark_object.y = filiter_list[index].y
      else:
        face_landmark_object.x = filiter_list[index][0]/width
        face_landmark_object.y = filiter_list[index][1]/height
         
    return face_landmark_backup


old_caloptflow = {
  "gray_src": None,
  "gray_pre": None,
  "height": 240,
  "width": 320,
}

def normalized_landmark(landmark, w, h):
  temp = []
  for i in landmark:
    temp.append( [i[0]/w, i[1]/h] )
  return temp

def low_pass(landmark_new, landmark_old):
  dx = landmark_new[0] - landmark_old[0]
  dy = landmark_new[1] - landmark_old[1]
  if( dx*dx + dy*dy > 0):
    return 1
  else:
    return 0

def smooth_point_by_caloptflow_old(face_landmark: list,  face_landmark_pre: list, width: int, height: int, name:str, frame):
  src = frame
  gray_src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
  gray_pre = old_caloptflow["gray_pre"]
  if(width != old_caloptflow["width"]):
    gray_src = cv2.resize(gray_src,(old_caloptflow["width"], old_caloptflow["height"]) )

  w,h = old_caloptflow["width"], old_caloptflow["height"]

  number_landmark = len(face_landmark)
  now_landmark = []
  pre_landmark = []

  diffs = []
  mean = 0
  for i, landmark_ in enumerate(face_landmark_pre):
    # p0.append([[landmark_[0][0], landmark_[0][1]]])
    old_point = copy.deepcopy( [[ landmark_.x*w, landmark_.y*h ]] )
    new_point = copy.deepcopy( [[ face_landmark[i].x*w, face_landmark[i].y*h ]] )
    pre_landmark.append(old_point)
    now_landmark.append(new_point)

    diff = math.sqrt( (old_point[0][0] - new_point[0][0])**2 + (old_point[0][1] - new_point[0][1])**2 )

    diffs.append(diff)
    mean += diff
  
  pre_landmark = np.array(pre_landmark, dtype = np.float32)
  now_landmark = np.array(now_landmark, dtype = np.float32)
  now_landmark_backup = copy.deepcopy(now_landmark)

  old_caloptflow["gray_pre"] = gray_src

  if(gray_pre is None):
    return normalized_landmark( np.squeeze(now_landmark), face_landmark, w, h)

  mean /= number_landmark
  variance = 0
  for diff_ in diffs:
    variance += (diff - mean)**2

  if(variance > 800):
  
    print("ignore big move");
    return normalized_landmark( np.squeeze(now_landmark), face_landmark, w, h)
  
  now_landmark, st, err = cv2.calcOpticalFlowPyrLK( gray_pre , gray_src, pre_landmark, now_landmark)
  _, _, rect_w, rect_h = cv2.boundingRect( np.squeeze( now_landmark ) )

  diffdlib = rect_h*rect_w /4000
  now_landmark = np.squeeze(now_landmark)

  if(variance/number_landmark < diffdlib):
    for index in range(number_landmark):
      if(not self.low_pass(now_landmark[index], pre_landmark[index])):
        now_landmark[index][0] = pre_landmark[index][0]/w 
        now_landmark[index][1] = pre_landmark[index][1]/h 

    return now_landmark 
  else:
    return np.squeeze( now_landmark_backup )
 
def save_ans_to_file(landmark_list, file_path = "./save/ans.txt", time_index = 0 ):
  landmark_x_list = []
  landmark_y_list = []
  f = open(file_path, 'a+')
  f.write(">"+ str(time_index)+"\n")
  if( len(landmark_list) <5):
    pass
  else:
    for landmark in landmark_list:
      #print(landmark)
      landmark_x_list.append(landmark.x)
      landmark_y_list.append(landmark.y)
      f.write(str(landmark)+",")
  f.close()
  
  
  

     

  
  

  

  


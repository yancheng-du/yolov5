import d3dshot
import numpy as np
import cv2
import time
import torch

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadWebcam, LoadScreen
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from torchvision import transforms

import win32api


#create screening capturing tool
d = d3dshot.create("numpy")
d.display = d.displays[0]
RGBtoBGR = [2, 1, 0]

#dimension of the mouse position
target_width = 1280
target_height = 720

v_scaling = target_height/384
h_scaling = target_width/640

screen_center_x = int(640/2)
screen_center_y = int(384/2)


if(torch.cuda.is_available()):
	device = select_device('0')
else:
	device = 'cpu'

p = transforms.Compose([transforms.Resize((384, 640))])
#model parameters
weights = 'yolov5s.pt'
dnn = False
data = 'data/coco128.yaml'
half = False
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
print("yolo model created")

# prediction parameters
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45 # NMS IOU threshold
max_det = 50
classes = 0	# only detecting people
agnostic_nms=False

current_time = 0
last_time = 0

move_speed = 0.1

n = 0
screen_shot_time = 0
preprocessing_time = 0
prediction_time = 0
post_processing_time = 0
mouse_move_time = 0



while True:
	n += 1
	im = d.screenshot()[:,:,RGBtoBGR]
	current_time = time.time()
	screen_shot_time += 1/n + (current_time-last_time - screen_shot_time)
	last_time = current_time
	
	im2 = torch.from_numpy(im).to(device)
	im2 = p(im2[None, :].permute([0,3,1,2]))/255
	#print(im2.shape)
	
	'''im = dataset.imgs[0]
	im = torch.from_numpy(im).to(device)
	im = im.half() if half else im.float()  # uint8 to fp16/32
	im /= 255  # 0 - 255 to 0.0 - 1.0
	if len(im.shape)==3:
		im = im[None]  # expand for batch dim
	'''
	current_time = time.time()
	preprocessing_time += 1/n + (current_time-last_time - preprocessing_time)
	last_time = current_time
	
	pred = model(im2, augment=False, visualize=False)
	#print(pred.shape)
	pred = non_max_suppression(pred,conf_thres,iou_thres,classes,agnostic_nms,max_det)
	
	current_time = time.time()
	prediction_time += 1/n + (current_time-last_time - prediction_time)
	last_time = current_time
	
	ox, oy = win32api.GetCursorPos()
	ox = int(ox/h_scaling)
	oy = int(oy/v_scaling)
	min_dist = np.inf
	min_loc = ()
	detect = False
	for objects in pred[0]:
		center_x = int(0.5*h_scaling*(objects[0]+objects[2]))
		center_y = int(0.5*v_scaling*(objects[1]+objects[3]))
		dist_from_mouse =(center_y-oy)**2 + (center_x-ox)**2
		if(dist_from_mouse< min_dist):
			min_dist = dist_from_mouse
			min_loc = (center_x, center_y)
			detect = True
	
	current_time = time.time()
	post_processing_time += 1/n + (current_time-last_time - post_processing_time)
	last_time = current_time
	
	if detect:
		#im_mat = cv2.circle(im_mat, min_loc, radius=5, color=(0, 0, 255), thickness=-1)
		move_speed = np.exp(-0.0001 * min_dist)
		target_x = int((ox + move_speed*(min_loc[0] - ox)) * h_scaling)
		target_y = int((oy + move_speed*(min_loc[1] - oy)) * v_scaling)
		win32api.SetCursorPos((target_x,target_y))
		
	current_time = time.time()
	mouse_move_time += 1/n + (current_time-last_time - mouse_move_time)
	last_time = current_time
	
	#cv2.putText(im_mat, "{}, {}".format(str(int(ox)), str(int(oy))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
	#current_time = time.time()
	#time_since_last = current_time - last_time
	#last_time = current_time
	#cv2.putText(im_mat, "FPS: {}".format(str(int(1/time_since_last))), (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 2, cv2.LINE_4)
	
	if cv2.waitKey(1)&0xFF==ord('q') or n>1000:
		break
		
print("screen_shot_time = {}\npre_processing_time = {}\nprediction_time = {}\npost_processing_time = {}\nmouse_move_time = {}".format(screen_shot_time,preprocessing_time,prediction_time,post_processing_time,mouse_move_time))
print('total_time = {}'.format(screen_shot_time+preprocessing_time+prediction_time+post_processing_time+mouse_move_time))
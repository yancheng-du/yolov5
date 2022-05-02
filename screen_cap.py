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

import win32api, win32con
import pyautogui
import mss


def cal_time(last_time, time_to_cal,n):
	current_time = time.time()
	ret = time_to_cal + 1/n*(current_time-last_time-time_to_cal)
	last_time = current_time
	return ret,last_time

def inblock(mouse_x, mouse_y, block, r=1):
	#print(block[0:4])
	#print(mouse_x,mouse_y)
	return 0.9*block[0]+0.1*block[2]< mouse_x < 0.9*block[2]+0.1*block[0] and block[1] < mouse_y < r*(block[3]-block[1])+block[1]

def main():
	
	dx = 0
	dy = 0
	
	screen_dim = (1440, 2560)  # in H * W
	screen_center = (screen_dim[0]//2, screen_dim[1]//2)
	cropping_size = (480, 640)  # in H * W
	crop_center = (cropping_size[0]//2, cropping_size[1]//2)
	screen_crop_loc = (screen_center[0]-cropping_size[0]//2, screen_center[1]-cropping_size[1]//2)  # in H * W
	
	
	if(torch.cuda.is_available()):
		device = select_device('0')
	else:
		device = 'cpu'
	
	#model parameters
	weights = 'yolov5s.pt'
	dnn = False
	data = 'data/coco128.yaml'
	half = False
	model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None)
	print("yolo model created")
	
	# prediction parameters
	conf_thres = 0.25  # confidence threshold
	iou_thres = 0.45 # NMS IOU threshold
	max_det = 50
	classes = 0	# only detecting people
	agnostic_nms=False
	move = True
	
	move_speed = 0.1
	
	n = 0
	screen_shot_time = 0
	preprocessing_time = 0
	prediction_time = 0
	post_processing_time = 0
	mouse_move_time = 0
	
	
	monitor = {"top": screen_crop_loc[0], "left": screen_crop_loc[1], "width": cropping_size[1], "height": cropping_size[0], "mon":1}
	with mss.mss() as sct:
		while True:
			n += 1
			last_time = time.time()
			im = np.array(sct.grab(monitor))
			#print(im.shape)
			#cv2.imshow("OpenCV/Numpy normal", im)
			
			screen_shot_time,last_time = cal_time(last_time, screen_shot_time, n)
			
			im2 = torch.from_numpy(im).to(device)
			im2 = im2[None, :, :, :3].permute([0,3,1,2])/255
			#print(im2.shape)
			
			preprocessing_time,last_time  = cal_time(last_time, preprocessing_time, n)
			
			pred = model(im2, augment=False, visualize=False)
			#print(pred.shape)
			pred = non_max_suppression(pred,conf_thres,iou_thres,classes,agnostic_nms,max_det)
			
			prediction_time,last_time  = cal_time(last_time, prediction_time, n)
			
			min_dist = np.inf
			min_loc = (0, 0)
			detect = False
			for objects in pred[0]:
				center_x = int(0.5*(objects[0]+objects[2]))
				center_y = int(0.5*objects[1]+0.5*objects[3])
				dist_from_mouse = np.linalg.norm([center_x-crop_center[1], center_y-crop_center[0]])
				if (dist_from_mouse<min_dist):
					min_dist = dist_from_mouse
					min_loc = (center_x, center_y)
					detect = True
			
			if move:
				if detect:
					move_speed = 1
					dx = np.clip(int(0.50*move_speed*(min_loc[0]-crop_center[1])+0.50*dy), -90, 90)
					# target_x = max(30,min(-30,target_x))
					dy = np.clip(int(0.50*move_speed*(min_loc[1]-crop_center[0])+0.50*dy), -30, 30)
					# target_y = max(30, min(-30, target_y))
					if(np.abs(min_loc[0]-crop_center[1]) < 5):
						dx = 0
					if(np.abs(min_loc[1]-crop_center[0]) < 5):
						dy = 0
					win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
			else:
				for object in pred[0]:
					if inblock(crop_center[1],crop_center[0],object):
						win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
						win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
				
			
			post_processing_time,last_time  = cal_time(last_time, post_processing_time, n)
			
			img_out = cv2.circle(im,min_loc,3,(255, 0, 0),2)
			cv2.imshow("OpenCV/Numpy normal", img_out)
			
			mouse_move_time,last_time  = cal_time(last_time, mouse_move_time, n)
			
			if cv2.waitKey(25)&0xFF==ord("t"):
				move = not move
			
			if cv2.waitKey(25)&0xFF==ord("q"):
				cv2.destroyAllWindows()
				break
			
	
	print("screen_shot_time = {}\npre_processing_time = {}\nprediction_time = {}\npost_processing_time = {}\nmouse_move_time = {}".format(screen_shot_time,preprocessing_time,prediction_time,post_processing_time,mouse_move_time))
	print('total_time = {}'.format(screen_shot_time+preprocessing_time+prediction_time+post_processing_time+mouse_move_time))
	
	
if __name__ == '__main__':
	main()
	

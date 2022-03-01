import d3dshot
import numpy as np
import cv2
import time
import torch

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadWebcam
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from torchvision import transforms


#create screening capturing tool
d = d3dshot.create("numpy")
d.display = d.displays[1]
RGBtoBGR = [2, 1, 0]

target_width = 1280
target_height = 720

v_scaling = target_height/384
h_scaling = target_width/640


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


while True:
	im = d.screenshot()[:,:,RGBtoBGR]
	im_mat = cv2.resize(im.copy(),(target_width,target_height))
	im2 = torch.from_numpy(im).to(device)
	im2 = p(im2[None, :].permute([0,3,1,2]))/255
	#print(im2.shape)
	pred = model(im2, augment=False, visualize=False)
	#print(pred.shape)
	pred = non_max_suppression(pred,conf_thres,iou_thres,classes,agnostic_nms,max_det)
	for objects in pred[0]:
		center_x = int(0.5*h_scaling*(objects[0]+objects[2]))
		center_y = int(0.5*v_scaling*(objects[1]+objects[3]))
		im_mat = cv2.circle(im_mat, (center_x,center_y), radius=5, color=(0, 0, 255), thickness=-1)

	cv2.imshow("screen capture", im_mat)
	
	if cv2.waitKey(1)&0xFF==ord('q'):
		break
		
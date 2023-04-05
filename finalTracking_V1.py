
# package import for fastmot

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import json
import cv2
import numpy

import fastmot
from fastmot import models
from fastmot.utils import ConfigDecoder, Profiler

# package import for yolov5 (waste detection)
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, 
                                  non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# package import for customer
from customer import Customer, getCustomerList, getDress, isDressSame
import abnormal

def yolorun(device, model, imgsz, frame):
	
	stride, names, pt = model.stride, model.names, model.pt
	# Dataloader for frame
	dataset = LoadImages(frame, img_size=imgsz, stride=stride, auto=pt)
	bs = 1 # batch_size
	vid_path, vid_writer = [None] * bs, [None] * bs

	# Run inference
	model.warmup(imgsz=(1 if pt else bs, 3, *imgsz)) # warmup ...? check 해봐야함
	seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
	for im, im0s, vid_cap, s in dataset:
		t1 = time_sync()
		im = torch.from_numpy(im).to(device)
		im = im.half() if model.fp16 else im.float()
		im /= 255 # normalization
		if len(im.shape) == 3:
			im = im[None]
		t2 = time_sync()
		dt[0] += t2 - t1

		# Inference
		pred = model(im, augment=False, visualize=False)
		t3 = time_sync()
		dt[1] += t3 - t2

		# NMS
		pred = non_max_suppression(pred, 0.7, 0.45, None, False, max_det=1000)
		dt[2] += time_sync() - t3
		
		# Process predictions
		xyxy_list = []
		for i, det in enumerate(pred):
			seen += 1
			# path는 쓰면서 생각해보자 -> path return안하도록 수정함
			im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
			s += '%gx%g ' % im.shape[2:]
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
			annotator = Annotator(im0, line_width=2, example=str(names))
			if len(det):
				det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum() # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					c = int(cls)
					label = (f'{names[c]} {conf:.2f}')
					annotator.box_label(xyxy, label, color=colors(c, True))
					xyxy_list.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), names[c]])
			return annotator.result(), xyxy_list

def main():
    
    # set up logging
	logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger("final") # final 이라는 이름의 Logger 생성
	
	# load config file
	with open(Path(__file__).parent /'cfg'/'mot.json') as cfg_file:
		config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))
	# JSON 형태의 file을 Python class로 변환

	logger.info('Creating VideoStream object...')
	# stream1: 입장 카메라 (yolo detect 필요없음) stream2: 매장 카메라
	stream1 = fastmot.VideoIO(config.resize_to, '/dev/video0', None, **vars(config.stream_cfg)) # resolution, frame_rate, buffer_size
	stream2 = fastmot.VideoIO(config.resize_to, '/dev/video2', None, **vars(config.stream_cfg))
	
	mot1 = None
	mot2 = None
	
	txt = None
	draw = False
	detections_1 = [] # for stream1
	detections_2 = [] # for stream2
	logger.info('Creating fastmot object...')
	# FastMOT 객체 생성 (Tracking + Detection)
	mot1 = fastmot.MOT(config.resize_to, detections_1, **vars(config.mot_cfg), draw=draw) # mot_cfg: yolo_detector에 대한 cfg값
	mot1.reset(stream1.cap_dt) # mot.step() 전 mot객체 초기화 -> cap_dt = frame간 시간 간격을 뜻함
	
	mot2 = fastmot.MOT(config.resize_to, detections_2, **vars(config.mot_cfg), draw=draw)
	mot2.reset(stream2.cap_dt)


	customerList_1 = {} # 현재 frame에 존재하는 customer 객체 저장 -> from video0
	customerList_2 = {} # 현재 frame에 존재하는 customer 객체 저장 -> from video2
	customerList_t = {} # 모든 customer 객체 저장

	clusterList = {}

	waste = []
	trashFlag = False
	cnt = 0

	# load model for yolo
	device = select_device('')
	model = DetectMultiBackend('./weights/best.pt', device=device, dnn=False, data='./data/taco.yaml', fp16=False)
	stride = model.stride
	imgsz = check_img_size((640, 640), s=stride) # check image size
	
	# Open window
	cv2.namedWindow('Video0', cv2.WINDOW_AUTOSIZE)
	# cv2.namedWindow('Video1', cv2.WINDOW_AUTOSIZE)
	# cv2.namedWindow('Video0', cv2.WINDOW_NORMAL)
	# cv2.namedWindow('Video1', cv2.WINDOW_NORMAL)
	logger.info('Start video capture...')
	stream1.start_capture()
	stream2.start_capture()
	index = 0

	# output 영상
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output = cv2.VideoWriter('./stay_output.mp4', fourcc, 30.0, (1792, 504))

	try:
		with Profiler('app') as prof:
			while cv2.getWindowProperty('Video0', 0) >=0:
				frame_1 = stream1.read()
				frame_2 = stream2.read()
				if frame_1 is None or frame_2 is None:
					break
				mot1.step(frame_1)
				mot2.step(frame_2)
				# 일단 여기서 가라로 customer 객체 생성하고 함수화 하자 -> customer객체 저장 성공
				# Customer(self, id_num, detection, timestamp, dresscode)	

				# frame별로 현재 존재하는 Customer list = customerList
				# customerList_1 갱신
				c1 = getCustomerList(mot1.tracker, mot1.detections, mot1.timestamps, '1')
				if c1: # customerList가 None이 아니면
					for key in customerList_1:
						customerList_1[key].__del__() # 이전 frame 객체 삭제
					customerList_1.clear()
					customerList_1 = c1
				if not c1 and mot1.frame_count % 30 == 0 :
					customerList_1.clear()

				# 예외처리
				
				#for key in customerList_1:
				#	if not customerList_t[key].isin:
				#		customerList_1[key].__del__()

				
				# customerList_2 갱신
				c2 = getCustomerList(mot2.tracker, mot2.detections, mot2.timestamps, '2')
				if c2:
					for key in customerList_2:
						if (customerList_2[key].isre == True) and key in c2:
							
							c2[key].isre = True
							c2[key].customerId = customerList_2[key].customerId
							c2[key].isin = True
							c2[key].timestamp = customerList_2[key].timestamp
							
						else: customerList_2[key].__del__() # 이전 frame 객체 삭제
					customerList_2 = c2
					
				# customerList_t에 새로운 Customer가 들어오면 드레스 코드 포함하여 저장
				# customerList_1 (video0)
				for key in customerList_1:
					if key in customerList_t: # 전체 List에 등록되어 있으면 dress추출 수행 안함 
						continue
					customer = customerList_1[key]
					customer.dresscode = getDress(frame_1, customer.center)
					customerList_t[key] = customerList_1[key]
					logger.info('Found:			person		%s', key)
				# customerList_2 (video2)
				for key in customerList_2:
					if key in customerList_t: # 전체 List에 등록되어 있으면 dress 중복 수행 안함
						continue
					customer = customerList_2[key] # 새로 검출된 고객
					customer.dresscode = getDress(frame_2, customer.center)
					
					# 여기서 중복 확인
					count = 0
					
					for k in customerList_t:
						# video0에서 검출된 고객리스트 중 없어진 고객
						if int(k[0]) == 1 and customerList_t[k].isin == False and customerList_t[k].isre == False:
							res = isDressSame(customerList_t[k].dresscode, customer.dresscode)
							if res: #만약 두 옷이 같다면?
								# customerList_2에 id를 변경
								customerList_2[key].customerId = customerList_t[k].customerId
								customerList_2[key].timestamp = customerList_t[k].timestamp
								customerList_t[k].isin = True
								customerList_2[key].isre = True
								customerList_t[k].isre = True
								customerList_t[k] = customerList_2[key]
								customerList_t[key] = customerList_2[key]
								logger.info('Reidentified in camera 2   		person		%d', customerList_2[key].customerId)
								logger.info('it\'s key is 			%s', key) 
								count = count + 1

								break
					if (count == 0):	 			
						customerList_t[key] = customerList_2[key]
						logger.info('Found:			person		%s', key)

				# 현재 frame에서 사라진 고객 객체를 customer 객체에 표시
				# 생각 해봐야함 한번
				
				for key in customerList_t:
					if key not in customerList_1:
						customerList_t[key].isin = False
					if key not in customerList_2:
						customerList_t[key].isin = False
					else: customerList_t[key].isin = True
		
					
				# cluster detection on video2
				if (mot2.frame_count % 4 == 0):
					if (len(list(customerList_2.keys())) > 1):
						abnormal.isCluster(customerList_2, customerList_t, clusterList, frame_2)
				
				clusterText = ''
				for key in clusterList:
					if clusterList[key].isClusterLast() == True:
						clusterText = 'Cluster Alert:  ' + str(clusterList[key].ids[0]) + ', ' + str(clusterList[key].ids[1])
				
				# frame을 yolo 입력으로 넣고 그려서 나온 frame에 mot.detections 그리기 
				out, xyxy_list = yolorun(device, model, imgsz, frame_2)
				# garbage detection bbox 추출 10초마다 판단
				
				if (mot2.frame_count % 150 == 0 or mot2.frame_count == 1):
					if (cnt%2 != 0 and cnt != 0):
						waste.append(xyxy_list)
						past = waste[0]
						present = waste[1]
						for trash_1 in past:
							for trash_2 in present:
								if trash_1[-1] == trash_2[-1]: # trash id가 일치
									res = abnormal.caculateIOU(trash_1[:4], trash_2[:4])
									if (res > 0.8): # 영역도 겹친다면?
										print('trash remained: alert', trash_2)
										trashText = 'trash remained alert'
										trashFlag = True

						waste = []
						cnt = cnt+1
					else:
						waste.append(xyxy_list)
						# print(waste)
						cnt = cnt+1

				mot1._draw(frame_1, mot1.detections, customerList_1)
				mot2._draw(out, mot2.detections, customerList_2)
				
				# print text in opencv window
				for key in clusterList:
					clusterList[key].draw(out)
				
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(out, clusterText, (700, 100), font, 1, (0, 0, 255), 2)
				if trashFlag:
					cv2.putText(out, trashText, (700, 100), font, 1, (0, 0, 255), 2)
				
				for key in customerList_t:
					if customerList_t[key].isCustomerStay() and customerList_t[key].isin:
						stayText = 'Alert: customer stay'
						cv2.putText(out, stayText, (700, 100), font, 1, (0, 0, 255), 2)

				result_ = cv2.hconcat([frame_1, out])
				result = cv2.resize(result_, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
				cv2.imshow('Video0', result)
				# cv2.imshow('Video1', out)
				output.write(result)
				if cv2.waitKey(1) & 0xFF == 27:
					break
	finally:
		# clean up resources
		stream1.release()
		stream2.release()
		output.release()	
		cv2.destroyAllWindows()
		print('result size: ', result.shape[:2])
	for key in customerList_t:
		print("id: ", customerList_t[key].customerId, "timestamp: ", customerList_t[key].timestamp)
		print("isin: " , customerList_t[key].isin, "key: ", key)
	
	for key in clusterList:
		print("cluster: ", clusterList[key])

	# FPS 계산
	avg_fps = round(mot1.frame_count / prof.duration)
	logger.info('Average FPS: %d', avg_fps)
	# Static Method
	mot1.print_timing_info()
	

if __name__ == "__main__":
    main()


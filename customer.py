
import time
import cv2
import numpy as np

def getCustomerList(tracker, detections, timestamps, videoNum):
	tracks = []
	customerList = {}
	if tracker.tracks:
		tracks = list(tracker.tracks.keys())
		cnt = 0
		
		if(len(detections) == 0 or type(detections) is list or len(tracks) == 0
			or not timestamps): pass
		else:
			for detection in detections:
				if len(tracks) <= cnt: break
				if tracks[cnt] in timestamps:
					customerList[videoNum+str(tracks[cnt])] = Customer(int(videoNum + str(tracks[cnt])), 
																detection.tlbr, timestamps[tracks[cnt]], None)
					cnt = cnt+1
	return customerList

def getDress(img, center):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	return hsv[center[0]-100:center[0]+100, center[1]-100:center[1]+100]

def isDressSame(img1, img2):
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
	cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])
	cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

	method = cv2.HISTCMP_BHATTACHARYYA
	ret = cv2.compareHist(hist1, hist2, method)
	# ret = 1이면 완전 일치
	if ret > 0.8:
		return True
	else:
		return False

class Customer:
	
	def __init__(self, id_num, detection, timestamp, dresscode):
		
		self.customerId = id_num
		self.tlbr = detection
		self.dresscode = dresscode
		self.timestamp = timestamp
		self.center = [int((self.tlbr[0] + self.tlbr[2])/2) , int((self.tlbr[1] + self.tlbr[3])/2)]

		# customer state
		self.isin = True
		self.isre = False
		self.isStay = False
		self.isCluster = False
		self.isAbnormal = False
		self.isWaste = False

	
	def __del__(self):
		pass
	
	def isCustomerStay(self):
		
		Nowtime = time.time()
		t = Nowtime - self.timestamp
		if (time.gmtime(t).tm_sec > 10) : # 15 sec thres
			self.isStay = True
			return True
		else: 
			self.isStay = False
			return False
			

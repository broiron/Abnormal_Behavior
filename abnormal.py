import time
import cv2
import numpy as np

def caculateIOU(box1, box2):
	# box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def isCluster(customerList, customerList_t, clusterList, img):
	
	# 해야할 일: 여기서 time변수 저장하고 판단+그림그리기, text 포함

	key = list(customerList.keys())
	c1, c2 = customerList[key[0]], customerList[key[1]]
	iou = caculateIOU(c1.tlbr, c2.tlbr)
	if iou > 0.1: # 이 Frame에서 검출된 cluster?
		if customerList_t[key[0]].isCluster or  customerList_t[key[1]].isCluster:
			# 이미 검출된 cluster라면
			c = clusterList[str(key[0]) + str(key[1])]
			c.bboxupdate(c1.tlbr, c2.tlbr)
			c.draw(img)
		else: # 새로운 cluster
			c = Cluster(c1.tlbr, c2.tlbr, [c1.customerId, c2.customerId])
			customerList_t[str(c1.customerId)].isCluster = True
			customerList_t[str(c2.customerId)].isCluster = True
			clusterList[str(c1.customerId) + str(c2.customerId)] = c
			c.draw(img)
def getBig(x1, x2):
	if x1 >= x2:
		return x1
	else: return x2

def getSmall(x1, x2):
	if x1 <= x2:
		return x1
	else: return x2

class Cluster: #detect.tlbr -> tl, br = tuple(tlbr[:2]), tuple(tlbr[2:])
							# detect.tlbr = [x1, y1, x2, y2]
	def __init__(self, box1, box2, ids):
		self.tl = [getSmall(box1[0], box2[0]), getSmall(box1[1], box2[1])]
		self.br = [getBig(box1[2], box2[2]), getBig(box1[3], box2[3])]
		self.box = tuple(self.tl + self.br)
		self.time = time.time()
		self.ids = ids # list of ids
	
	def bboxupdate(self, box1, box2):
		self.tl = [getSmall(box1[0], box2[0]), getSmall(box1[1], box2[1])]
		self.br = [getBig(box1[2], box2[2]), getBig(box1[3], box2[3])]
		self.box = tuple(self.tl + self.br)

	def draw(self, img):
		p1, p2 = (int(self.tl[0]), int(self.tl[1])), (int(self.br[0]), int(self.br[1]))
		cv2.rectangle(img, p1, p2, (240, 0, 0), thickness=1, lineType=cv2.LINE_AA)

	def getBig(self, x1, x2):
		if x1 >= x2:
			return x1
		else: return x2
	
	def getSmall(self, x1, x2):
		if x1 <= x2:
			return x1
		else: return x2
	
	def isClusterLast(self):
		t = time.time() - self.time
		if (time.gmtime(t).tm_sec > 10) : # 1 minutes thres
			return True
		else: return False
	
	def deleteCluster(self):
		if not self.ids:
			__del__()


def isStay(customerList):
	
	for customer in customerList:
		t = time.time() - timcutomerList[customer].timestamp
		if (time.gmtime(t).tm_sec > 10):
			customer.isStay = True
			return True

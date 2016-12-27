import tfdeploy as td
import numpy as np

import sys
import os
import cv2
from scipy import misc
from uuid import uuid4

class MTCNN_tfdeploy:
	def __init__(self, model_path="../../data/"):		
		pnet_model = td.Model(os.path.join(model_path, "pnet.pkl"))
		rnet_model = td.Model(os.path.join(model_path, "rnet.pkl"))
		onet_model = td.Model(os.path.join(model_path, "onet.pkl"))
		
		self.p_in, self.p_out0, self.p_out1 = pnet_model.get('pnet/input:0', 'pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0')
		#self.p_in, self.p_out1 = pnet_model.get('pnet/input:0', 'pnet/prob1:0')
		self.r_in, self.r_out0, self.r_out1 = rnet_model.get('rnet/input:0', 'rnet/conv5-2/conv5-2:0', 'rnet/prob1:0')
		self.o_in, self.o_out0, self.o_out1, self.o_out2 = onet_model.get('onet/input:0', 'onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0')
		
	def detect_face(self, img, minsize = 20, threshold = [ 0.6, 0.7, 0.7 ], factor = 0.709):
		# im: input image
		# minsize: minimum of faces' size
		# pnet, rnet, onet: caffemodel
		# threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
		# fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
		factor_count=0
		total_boxes=np.empty((0,9))
		points=[]
		h=img.shape[0]
		w=img.shape[1]		
		minl=np.amin([h, w])
		m=12.0/minsize
		minl=minl*m
		# creat scale pyramid
		scales=[]
		while minl>=12:
			scales += [m*np.power(factor, factor_count)]
			minl = minl*factor
			factor_count += 1

		# first stage
		for j in range(len(scales)):
			scale=scales[j]
			hs=int(np.ceil(h*scale))
			ws=int(np.ceil(w*scale))
			im_data = self.imresample(img, (hs, ws))
			im_data = (im_data-127.5)*0.0078125
			img_x = np.expand_dims(im_data, 0)
			img_y = np.transpose(img_x, (0,2,1,3))
			#out = pnet(img_y)
			#out0 = np.transpose(out[0], (0,2,1,3))
			#out1 = np.transpose(out[1], (0,2,1,3))			
			uuid = uuid4()
			out0 = np.transpose(self.p_out0.eval({self.p_in: img_y}, uuid), (0,2,1,3))
			out1 = np.transpose(self.p_out1.eval({self.p_in: img_y}, uuid), (0,2,1,3))
			
			boxes, _ = self.generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])
			
			# inter-scale nms
			pick = self.nms(boxes.copy(), 0.5, 'Union')
			if boxes.size>0 and pick.size>0:
				boxes = boxes[pick,:]
				total_boxes = np.append(total_boxes, boxes, axis=0)

		numbox = total_boxes.shape[0]
		if numbox>0:
			pick = self.nms(total_boxes.copy(), 0.7, 'Union')
			total_boxes = total_boxes[pick,:]
			regw = total_boxes[:,2]-total_boxes[:,0]
			regh = total_boxes[:,3]-total_boxes[:,1]
			qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
			qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
			qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
			qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
			total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
			total_boxes = self.rerec(total_boxes.copy())
			total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
			dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)

		numbox = total_boxes.shape[0]
		if numbox>0:
			# second stage
			tempimg = np.zeros((24,24,3,numbox))
			for k in range(0,numbox):
				tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
				tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
				if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
					tempimg[:,:,:,k] = self.imresample(tmp, (24, 24))
				else:
					return np.empty()
			tempimg = (tempimg-127.5)*0.0078125
			tempimg1 = np.transpose(tempimg, (3,1,0,2))
			#out = rnet(tempimg1)			
			#out0 = np.transpose(out[0])
			#out1 = np.transpose(out[1])
			uuid = uuid4()
			out0 = np.transpose(self.r_out0.eval({self.r_in: tempimg1}, uuid))
			out1 = np.transpose(self.r_out1.eval({self.r_in: tempimg1}, uuid))
			score = out1[1,:]
			ipass = np.where(score>threshold[1])
			total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
			mv = out0[:,ipass[0]]
			if total_boxes.shape[0]>0:
				pick = self.nms(total_boxes, 0.7, 'Union')
				total_boxes = total_boxes[pick,:]
				total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
				total_boxes = self.rerec(total_boxes.copy())

		numbox = total_boxes.shape[0]
		if numbox>0:
			# third stage
			total_boxes = np.fix(total_boxes).astype(np.int32)
			dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)
			tempimg = np.zeros((48,48,3,numbox))
			for k in range(0,numbox):
				tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
				tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
				if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
					tempimg[:,:,:,k] = self.imresample(tmp, (48, 48))
				else:
					return np.empty()
			tempimg = (tempimg-127.5)*0.0078125
			tempimg1 = np.transpose(tempimg, (3,1,0,2))
			#out = onet(tempimg1)
			#out1 = np.transpose(out[1])
			#out2 = np.transpose(out[2])
			uuid = uuid4()
			out0 = np.transpose(self.o_out0.eval({self.o_in: tempimg1}, uuid))
			out1 = np.transpose(self.o_out1.eval({self.o_in: tempimg1}, uuid))
			out2 = np.transpose(self.o_out2.eval({self.o_in: tempimg1}, uuid))
			score = out2[1,:]
			points = out1
			ipass = np.where(score>threshold[2])
			points = points[:,ipass[0]]
			total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
			mv = out0[:,ipass[0]]

			w = total_boxes[:,2]-total_boxes[:,0]+1
			h = total_boxes[:,3]-total_boxes[:,1]+1
			points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
			points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
			if total_boxes.shape[0]>0:
				total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv))
				pick = self.nms(total_boxes.copy(), 0.7, 'Min')
				total_boxes = total_boxes[pick,:]
				points = points[:,pick]
					
		return total_boxes, points

	# function [boundingbox] = bbreg(boundingbox,reg)
	def bbreg(self,boundingbox,reg):
		# calibrate bounding boxes
		if reg.shape[1]==1:
			reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

		w = boundingbox[:,2]-boundingbox[:,0]+1
		h = boundingbox[:,3]-boundingbox[:,1]+1
		b1 = boundingbox[:,0]+reg[:,0]*w
		b2 = boundingbox[:,1]+reg[:,1]*h
		b3 = boundingbox[:,2]+reg[:,2]*w
		b4 = boundingbox[:,3]+reg[:,3]*h
		boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
		return boundingbox
		
	def generateBoundingBox(self, imap, reg, scale, t):
		# use heatmap to generate bounding boxes
		stride=2
		cellsize=12

		imap = np.transpose(imap)
		dx1 = np.transpose(reg[:,:,0])
		dy1 = np.transpose(reg[:,:,1])
		dx2 = np.transpose(reg[:,:,2])
		dy2 = np.transpose(reg[:,:,3])
		y, x = np.where(imap >= t)
		if y.shape[0]==1:
			dx1 = np.flipud(dx1)
			dy1 = np.flipud(dy1)
			dx2 = np.flipud(dx2)
			dy2 = np.flipud(dy2)
		score = imap[(y,x)]
		reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
		if reg.size==0:
			reg = np.empty((0,3))
		bb = np.transpose(np.vstack([y,x]))
		q1 = np.fix((stride*bb+1)/scale)
		q2 = np.fix((stride*bb+cellsize-1+1)/scale)
		boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
		return boundingbox, reg
	
	# function pick = nms(boxes,threshold,type)
	def nms(self, boxes, threshold, method):
		if boxes.size==0:
			return np.empty((0,3))
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
		s = boxes[:,4]
		area = (x2-x1+1) * (y2-y1+1)
		I = np.argsort(s)
		pick = np.zeros_like(s, dtype=np.int16)
		counter = 0
		while I.size>0:
			i = I[-1]
			pick[counter] = i
			counter += 1
			idx = I[0:-1]
			xx1 = np.maximum(x1[i], x1[idx])
			yy1 = np.maximum(y1[i], y1[idx])
			xx2 = np.minimum(x2[i], x2[idx])
			yy2 = np.minimum(y2[i], y2[idx])
			w = np.maximum(0.0, xx2-xx1+1)
			h = np.maximum(0.0, yy2-yy1+1)
			inter = w * h
			if method is 'Min':
				o = inter / np.minimum(area[i], area[idx])
			else:
				o = inter / (area[i] + area[idx] - inter)
			I = I[np.where(o<=threshold)]
		pick = pick[0:counter]
		return pick

	# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
	def pad(self, total_boxes, w, h):
		# compute the padding coordinates (pad the bounding boxes to square)
		tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
		tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
		numbox = total_boxes.shape[0]

		dx = np.ones((numbox), dtype=np.int32)
		dy = np.ones((numbox), dtype=np.int32)
		edx = tmpw.copy().astype(np.int32)
		edy = tmph.copy().astype(np.int32)

		x = total_boxes[:,0].copy().astype(np.int32)
		y = total_boxes[:,1].copy().astype(np.int32)
		ex = total_boxes[:,2].copy().astype(np.int32)
		ey = total_boxes[:,3].copy().astype(np.int32)

		tmp = np.where(ex>w)
		edx[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
		ex[tmp] = w
		
		tmp = np.where(ey>h)
		edy[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
		ey[tmp] = h

		tmp = np.where(x<1)
		dx[tmp] = np.expand_dims(2-x[tmp],1)
		x[tmp] = 1

		tmp = np.where(y<1)
		dy[tmp] = np.expand_dims(2-y[tmp],1)
		y[tmp] = 1
		
		return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph
		
	# function [bboxA] = rerec(bboxA)
	def rerec(self, bboxA):
		# convert bboxA to square
		h = bboxA[:,3]-bboxA[:,1]
		w = bboxA[:,2]-bboxA[:,0]
		l = np.maximum(w, h)
		bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
		bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
		bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
		return bboxA		
		
	def imresample(self, img, sz):
		im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #pylint: disable=no-member
		return im_data

		
if __name__ == "__main__":
	img = misc.imread(sys.argv[1])
	
	mtcnn = MTCNN_tfdeploy()
	bounding_boxes, points = mtcnn.detect_face(img)
	
	print(bounding_boxes, points)
	
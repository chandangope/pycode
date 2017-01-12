import os.path
import sys
import math
import time
import tensorflow as tf
import numpy as np
import cv2
import Tkinter
import tkFileDialog

sys.path.append('../../myutils')

import Utils

_utils = Utils.Utils()

resizedImgRows = 72
resizedImgCols = 36

frameResizeRows = 360
frameResizeCols = 640


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'trainedmodel', 'Directory with trained model')
flags.DEFINE_string('detections_dump_dir', 'detections_dump', 'Directory for dumping detections')
flags.DEFINE_string('videos_folder', '/home/ivusi7dl/Videos/Negative', 'Folder with negative videos')
flags.DEFINE_string('dump_detections', 'TRUE', 'Whether to save detected rois')


def weight_variable(shape, varname):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=varname)

def bias_variable(shape, varname):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=varname)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def countFiles(path):
	root, dirs, files = os.walk(path)
	return len(files)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def getROIs():
	rois = []

	w = 24
	h = 48
	x = 200
	y = 145
	while x + w <  frameResizeCols-200:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/2


	w = 36
	h = 72
	x = 160
	y = 135 #140
	while x + w <  frameResizeCols-160:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/3

	w = 40
	h = 80
	x = 0
	y = 140 #140
	while x <  frameResizeCols-w:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/2

	w = 48
	h = 96
	x = 0
	y = 140
	while x <  frameResizeCols-w:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/3

	w = 80
	h = 160
	x = 0
	y = 130
	while x <  frameResizeCols-w:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/3


	w = 80
	h = 160
	x = 0
	y = 100
	while x <  frameResizeCols-w:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4





	w = 36
	h = 72
	x = 0
	y = 60
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 120
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 140
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 160
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 180
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 200
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 0
	y = 220
	while x + w <  frameResizeCols-36:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 48
	h = 96
	x = 0
	y = 160
	while x + w <  frameResizeCols-48:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 72
	h = 144
	x = 0
	y = 120
	while x + w <  frameResizeCols-72:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 72
	h = 144
	x = 0
	y = 140
	while x + w <  frameResizeCols-72:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 24
	h = 48
	x = 0
	y = 160
	while x + w <  frameResizeCols-24:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	return rois
    
#OPENCV RESIZE - columns first, then rows.
def classifyROI(roi, tfsess):
	res = cv2.resize(roi,(resizedImgCols, resizedImgRows), interpolation = cv2.INTER_CUBIC)
	res = res/255.0
	#m,s = cv2.meanStdDev(res)
	#s = max(s, 1.0/math.sqrt(res.size))
	#print ("res dtype before = {0}".format(res.dtype))
	#res = (res-m)
	#print ("res dtype = {0}".format(res.dtype))

	resfloat = res.astype(dtype="float32")
	#print ("resfloat dtype = {0}".format(resfloat.dtype))
	
	testimage = resfloat.reshape(-1,resizedImgRows*resizedImgCols)
	res = tfsess.run(y_conv, feed_dict={x: testimage, keep_prob: 1.0})
	index = np.argmax(res)
	prob = res[0, index]
	return index, prob

#OPENCV RESIZE - columns first, then rows.
def classifyROIs(rois, tfsess):
	testimages = np.empty((0, resizedImgRows*resizedImgCols), dtype="float32")
	for roi in rois:
		res = cv2.resize(roi,(resizedImgCols, resizedImgRows), interpolation = cv2.INTER_CUBIC)
		res = res/255.0
		#m,s = cv2.meanStdDev(res)
		#s = max(s, 1.0/math.sqrt(res.size))
		#print ("res dtype before = {0}".format(res.dtype))
		#res = (res-m)
		#print ("res dtype = {0}".format(res.dtype))

		resfloat = res.astype(dtype="float32")
		#print ("resfloat dtype = {0}".format(resfloat.dtype))
	
		testimage = resfloat.reshape(-1,resizedImgRows*resizedImgCols)
		testimages = np.append(testimages, testimage, axis=0)

	results = tfsess.run(y_conv, feed_dict={x: testimages, keep_prob: 1.0})
	return results
	

#OPENCV RESIZE - columns first, then rows.
def saveImage(path, img):
	numFiles = _utils.countFiles(path)
	print("Num files in folder {0} = {1}".format(path, numFiles))
	imgresized = cv2.resize(img,(resizedImgCols, resizedImgRows), interpolation = cv2.INTER_CUBIC)
	cv2.imwrite(path + "/" + _utils.getDate() + "_" + _utils.getTime() + "_" + str(numFiles+1) + ".jpg", imgresized)
	


x = tf.placeholder("float", shape=[None, resizedImgRows*resizedImgCols])
x_image = tf.reshape(x, [-1,resizedImgRows, resizedImgCols,1])

Layer1_Filters = 24
Layer2_Filters = Layer1_Filters*2
Layer3_Filters = Layer2_Filters*2
FC1_nodes = Layer3_Filters*2

W_conv1 = weight_variable([3, 3, 1, Layer1_Filters], 'W_conv1')
b_conv1 = bias_variable([Layer1_Filters], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #36x72 will become 18x36 after 1st pooling

W_conv2 = weight_variable([3, 3, Layer1_Filters, Layer2_Filters], 'W_conv2')
b_conv2 = bias_variable([Layer2_Filters], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #18x36 will become 9x18 after 2nd pooling

W_conv3 = weight_variable([3, 3, Layer2_Filters, Layer3_Filters], 'W_conv3')
b_conv3 = bias_variable([Layer3_Filters], 'b_conv3')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv3_flat = tf.reshape(h_conv3, [-1, 9*18*Layer3_Filters])

W_fc1 = weight_variable([9*18*Layer3_Filters, FC1_nodes], 'W_fc1')
b_fc1 = bias_variable([FC1_nodes], 'b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([FC1_nodes, 2], 'W_fc2')
b_fc2 = bias_variable([2], 'b_fc2')
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


with tf.Session() as sess:
	# Restore variables from disk
	# Add ops to restore all the variables.
	saver = tf.train.Saver()

	checkpoint_file = os.path.join(FLAGS.train_dir, 'model.ckpt')
	saver.restore(sess, checkpoint_file)
	#print(sess.run(tf.all_variables()))
	print("Model loaded.")
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	videofiles = _utils.getAllFilesInDir(FLAGS.videos_folder)
	print("Num videos={0}".format(len(videofiles)))
	for f in videofiles:
		cap = cv2.VideoCapture(f)
	
		frameCount = 0
		totalTime = 0
		fps = "0"
		ret,frame = cap.read()
		height, width, channels = frame.shape
		print("File: {0}".format(f))
		print("Orig Frame height = {0}, Orig Frame width = {1}".format(height, width))
		confThresh = 0.75
		while(cap.isOpened()):
			ret,frame = cap.read()
			frameCount = frameCount+1
			if(frameCount==30):
				#print("frameCount=30, totalTime={0}".format(totalTime))
				fps = "fps=" + str(int(frameCount/totalTime))
				frameCount = 0
				totalTime = 0
			
			if(ret==True):
				grayorig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray = cv2.resize(grayorig,(frameResizeCols, frameResizeRows), interpolation = cv2.INTER_CUBIC)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			
				start_time = time.time()
				rois = getROIs()
				roiimgs = []
				for r in rois:
					x1 = r['x']
					y1 = r['y']
					x2 = x1+r['width']
					y2 = y1+r['height']
					roiimg = gray[y1:y2, x1:x2]
					roiimgs.append(roiimg)
				results = classifyROIs(roiimgs, sess)
				iter = 0
				for r in results:
					if(r[1] > confThresh):
						x1 = rois[iter]['x']
						y1 = rois[iter]['y']
						x2 = x1+rois[iter]['width']
						y2 = y1+rois[iter]['height']
						roiimg = gray[y1:y2, x1:x2]
						if(FLAGS.dump_detections ==  "TRUE"):
							saveImage(FLAGS.detections_dump_dir, roiimg)
					iter = iter+1

				# rois = getROIs()
				# for r in rois:
				# 	x1 = r['x']
				# 	y1 = r['y']
				# 	x2 = x1+r['width']
				# 	y2 = y1+r['height']
				# 	roiimg = gray[y1:y2, x1:x2]
				# 	index, prob = classifyROI(roiimg, sess)
				# 	if(index==1 and prob > confThresh):
				# 		if(FLAGS.dump_detections ==  "TRUE"):
				# 			saveImage(FLAGS.detections_dump_dir, roiimg)
			
				totalTime = totalTime + time.time()-start_time
				#print("totalTime={0}".format(totalTime))
					
				cv2.putText(gray,fps,(50,80), font, 0.8, (0,0,0), 2)
				cv2.imshow('frameresized',gray)
				
			else:
				break
			
		#
		cap.release()
		cv2.destroyAllWindows()
   

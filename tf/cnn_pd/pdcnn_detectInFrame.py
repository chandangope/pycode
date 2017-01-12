import os.path
import time
import tensorflow as tf
import numpy as np
import cv2
import Tkinter
import tkFileDialog
import sys

sys.path.append('../../myutils')

import Utils

_utils = Utils.Utils()


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'trainedmodel', 'Directory with trained model')

resizedImgRows = 72
resizedImgCols = 36

frameResizeRows = 360
frameResizeCols = 640


def weight_variable(shape, varname):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=varname)

def bias_variable(shape, varname):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=varname)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
	#print ("res = {0}".format(res))
	index = np.argmax(res)
	prob = res[0, index]
	return index, prob


def getROIs():
	rois = []

	w = 72
	h = 144
	x = 20
	y = 120
	while x + w <  frameResizeCols:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	w = 36
	h = 72
	x = 20
	y = 140
	while x + w <  frameResizeCols:
		roi =  {'x':x, 'y':y, 'width':w, 'height':h}
		rois.append(roi)
		x = x+w/4

	return rois

x = tf.placeholder("float", shape=[None, resizedImgRows*resizedImgCols])
x_image = tf.reshape(x, [-1,resizedImgRows, resizedImgCols,1])

Layer1_Filters = 16
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
	with tf.device("/cpu:0"):
		saver = tf.train.Saver()

		checkpoint_file = os.path.join(FLAGS.train_dir, 'model.ckpt')
		saver.restore(sess, checkpoint_file)
		#print(sess.run(tf.all_variables()))
		print("Model loaded.")
		
		Tkinter.Tk().withdraw() # Close the root window
		while(True):
			in_path = tkFileDialog.askopenfilename(initialdir='/home/ivusi7dl/Users/chandan/pycode/tf/generatesigs/testimages')
			if not in_path:
				break
			else:
				grayorig = cv2.imread(in_path,0)
				gray = cv2.resize(grayorig,(frameResizeCols, frameResizeRows), interpolation = cv2.INTER_CUBIC)
				rois = getROIs()
				for r in rois:
					x1 = r['x']
					y1 = r['y']
					x2 = x1+r['width']
					y2 = y1+r['height']
					roiimg = gray[y1:y2, x1:x2]
					index, prob = classifyROI(roiimg, sess)
					if(index==1 and prob > 0.50):
						cv2.rectangle(gray, (x1,y1), (x2,y2), (0,0,0), 2)
						text = "conf=" + str(int(prob*100))
						font = cv2.FONT_HERSHEY_SIMPLEX
						cv2.putText(gray, text, (x1,y1), font, 0.8, (0,0,0), 2)
				
				cv2.imshow('frame',gray)
				cv2.waitKey(0)
import numpy as np
import cv2
import os
import math
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('BaseFolder', '/home/ivusi7dl/Users/chandan/data/Nov23_2016', 'Base folder for data')
flags.DEFINE_string('TrainORValidate', 'Train', 'Whether to use train folder or validate folder')

resizedImgRows = 72
resizedImgCols = 36


#BatchNumber = 0

def processFolder(folderName):
	label = os.path.basename(folderName)
	label = np.asarray([int(label)], dtype="uint8")
	#print("label = {0}".format(label))
	onlyfiles = [os.path.join(folderName, f) for f in os.listdir(folderName) if os.path.isfile(os.path.join(folderName, f))]
	print ("numfiles = {0}".format(len(onlyfiles)))
	returndata = np.empty((0, resizedImgRows*resizedImgCols + 1), dtype="uint8")
	row = 0
	for f in onlyfiles:
		sig = getImgSig(f)
		label_plus_sig = np.concatenate([label, sig]).reshape(1,-1)
		returndata = np.append(returndata, label_plus_sig, axis=0)
	
	#print("returndata = \n{0}".format(returndata))
	return returndata
	

#OPENCV RESIZE - columns first, then rows.
def getImgSig(fileName):
	img = cv2.imread(fileName,0)
	#resized_img = cv2.resize(img,(resizedImgCols, resizedImgRows), interpolation = cv2.INTER_CUBIC)
	#m,s = cv2.meanStdDev(resized_img)
	#s = max(s, 1.0/math.sqrt(resized_img.size))
	#resized_img = (resized_img-m)/s
	#resized_img = resized_img/255.0
	#print ("min = {0}, max = {1}".format(resized_img.min(), resized_img.max()))
	#print ("resized_img dtype = {0}".format(resized_img.dtype))
	#resized_img = resized_img.astype(dtype="float32")
	#print ("resized_img dtype = {0}".format(resized_img.dtype))
	
	#return resized_img.flatten()
	return img.flatten()
	
	
	

if __name__ == '__main__':

	folderName = os.path.join(FLAGS.BaseFolder, FLAGS.TrainORValidate, '1')
	print("\nProcessing folder: {0}".format(folderName))
	data_1 = processFolder(folderName)
	print("data_1 shape = {0}, type = {1}".format(data_1.shape, data_1.dtype))
	
	
	folderName = os.path.join(FLAGS.BaseFolder, FLAGS.TrainORValidate, '0')
	print("\nProcessing folder: {0}".format(folderName))
	data_0 = processFolder(folderName)
	print("data_0 shape = {0}, type = {1}".format(data_0.shape, data_0.dtype))
	
	alldata = np.concatenate([data_0, data_1])
	print("\nalldata shape = {0}, type = {1}".format(alldata.shape, alldata.dtype))
	
	np.random.shuffle(alldata)
	print("Items with class 1 = {0}".format((alldata[:,0]==1).sum()))
	print("Items with class 0 = {0}".format((alldata[:,0]==0).sum()))
	
	oufile = FLAGS.BaseFolder + "/" + FLAGS.TrainORValidate + ".bin"
	f = open(oufile, 'wb')
	f.write(alldata)
	f.close()
	print("Signatures written to = {0}".format(oufile))
	


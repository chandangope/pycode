import os.path
import tensorflow as tf
import numpy as np
import time


start_time = time.time()


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 4e-5, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 170000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('checkpoint_steps', 5000, 'checkpoint steps')
flags.DEFINE_string('train_dir', 'trainedmodel', 'Directory to keep trained model')
flags.DEFINE_string('data_dir3', '/home/ivusi7dl/Users/chandan/data/Nov23_2016', 'Directory with Train.bin and Validate.bin')
flags.DEFINE_string('data_dir2', '/home/ivusi7dl/Users/chandan/data/Nov15_2016', 'Directory with Train.bin and Validate.bin')
flags.DEFINE_string('data_dir1', '/home/ivusi7dl/Users/chandan/data/Nov18_2016', 'Directory with Train.bin and Validate.bin')
flags.DEFINE_integer('printaccuracy_steps', 5000, 'Print accuracy after so many steps')

NUM_THREADS = 4

resizedImgRows = 72
resizedImgCols = 36

BatchNumber = 0
TrainDataFolders = [FLAGS.data_dir1, FLAGS.data_dir2, FLAGS.data_dir3];
TrainDataFoldersindex = 0;

f = open(TrainDataFolders[TrainDataFoldersindex] + '/Train.bin', 'rb')
print("Processing trg folder:{0}".format(TrainDataFolders[TrainDataFoldersindex]))
data_train = np.fromstring(f.read(), dtype="uint8")
data_train = data_train.reshape(-1,resizedImgRows*resizedImgCols+1)
print("data_train size:{0}".format(data_train.shape))
f.close()

# f2 = open(FLAGS.data_dir2 + '/Train.bin', 'rb')
# data_train2 = np.fromstring(f2.read(), dtype="uint8")
# data_train2 = data_train2.reshape(-1,resizedImgRows*resizedImgCols+1)
# print("data_train2 size:{0}".format(data_train2.shape))
# f2.close()

# f3 = open(FLAGS.data_dir3 + '/Train.bin', 'rb')
# data_train3 = np.fromstring(f3.read(), dtype="uint8")
# data_train3 = data_train3.reshape(-1,resizedImgRows*resizedImgCols+1)
# print("data_train3 size:{0}".format(data_train3.shape))
# f3.close()

# data_train = np.concatenate([data_train1, data_train2, data_train3])
# np.random.shuffle(data_train)
# print("data_train size:{0}".format(data_train.shape))


# f1 = open(FLAGS.data_dir1 + '/Validate.bin', 'rb')
# data_validate1 = np.fromstring(f1.read(), dtype="uint8")
# data_validate1 = data_validate1.reshape(-1,resizedImgRows*resizedImgCols+1)
# f1.close()

# f2 = open(FLAGS.data_dir2 + '/Validate.bin', 'rb')
# data_validate2 = np.fromstring(f2.read(), dtype="uint8")
# data_validate2 = data_validate2.reshape(-1,resizedImgRows*resizedImgCols+1)
# f2.close()

f3 = open(FLAGS.data_dir3 + '/Validate.bin', 'rb')
data_validate3 = np.fromstring(f3.read(), dtype="uint8")
data_validate3 = data_validate3.reshape(-1,resizedImgRows*resizedImgCols+1)
f3.close()

data_validate = np.concatenate([data_validate3])
#data_validate = np.concatenate([data_validate1, data_validate2, data_validate3])
print("data_validate size:{0}".format(data_validate.shape))


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
    
    
    
def convert2OneHot(y):
	rows = y.shape[0]
	y_onehot = np.zeros(shape=(rows,2))
	for i in xrange(y.shape[0]):
		if(y[i]==0):
			y_onehot[i,0]=1
			y_onehot[i,1]=0
		else:
			y_onehot[i,0]=0
			y_onehot[i,1]=1
			
	return y_onehot
	
	
def nextBatch(batchSize):
	global BatchNumber
	global data_train
	global TrainDataFoldersindex
	#print("BatchNumber: {0}".format(BatchNumber))
	startRow = batchSize*BatchNumber
	endRow = startRow + batchSize
	
	if(startRow > data_train.shape[0] or endRow > data_train.shape[0]):
		TrainDataFoldersindex = (TrainDataFoldersindex + 1)%len(TrainDataFolders)
		print("Processing trg folder:{0}".format(TrainDataFolders[TrainDataFoldersindex]))
		f = open(TrainDataFolders[TrainDataFoldersindex] + '/Train.bin', 'rb')
		data_train = np.fromstring(f.read(), dtype="uint8")
		data_train = data_train.reshape(-1,resizedImgRows*resizedImgCols+1)
		print("data_train size:{0}".format(data_train.shape))
		f.close()

		np.random.shuffle(data_train)
		BatchNumber = 0
		startRow = batchSize*BatchNumber
		endRow = startRow + batchSize
		
	features = data_train[startRow:endRow, 1:resizedImgRows*resizedImgCols+1]
	features = features.astype(dtype="float32")
	features = features/255.0

	#features = features - features.mean(axis=1, keepdims=True)  # subtract the mean of each row
	#m,s = cv2.meanStdDev(features)
	#s = max(s, 1.0/math.sqrt(resized_img.size))
	#features = (features-m)

	labels = data_train[startRow:endRow, 0]
	labels_onehot = convert2OneHot(labels)
	labels_onehot = labels_onehot.astype(dtype="float32")
	BatchNumber = BatchNumber+1
	return features, labels_onehot
	
	
def getValidationData():
	global data_validate
	startRow = 0
	endRow = data_validate.shape[0]
		
	features = data_validate[startRow:endRow, 1:resizedImgRows*resizedImgCols+1]
	features = features.astype(dtype="float32")
	features = features/255.0

	#features = features - features.mean(axis=1, keepdims=True)  # subtract the mean of each row
	#m,s = cv2.meanStdDev(features)
	#s = max(s, 1.0/math.sqrt(resized_img.size))
	#features = (features-m)

	labels = data_validate[startRow:endRow, 0]
	labels_onehot = convert2OneHot(labels)
	labels_onehot = labels_onehot.astype(dtype="float32")
	return features, labels_onehot



x = tf.placeholder("float", shape=[None, resizedImgRows*resizedImgCols])
y_ = tf.placeholder("float", shape=[None, 2])

x_image = tf.reshape(x, [-1,resizedImgRows,resizedImgCols,1])
print("shape of x_image:{0}".format(x_image.get_shape()))

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

reduct = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.reduce_mean(reduct)
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS, intra_op_parallelism_threads=NUM_THREADS))
sess = tf.Session() #by default it will use all cores

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2])

features_validation, labels_validation = getValidationData()
print("validate features size:{0}".format(features_validation.shape))
print("validate labels size:{0}".format(labels_validation.shape))


for i in range(FLAGS.max_steps):
    #batch = mnist.train.next_batch(FLAGS.batch_size) #each batch[0] is of shape (batch_size,784) and batch[1] is of shape (batch_size, 10)
    #sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    features, labels = nextBatch(FLAGS.batch_size)
    sess.run(train_step,feed_dict={x: features, y_: labels, keep_prob: 0.5})
    
    if(i + 1) % FLAGS.checkpoint_steps == 0:
	validate_accuracy = sess.run( accuracy, feed_dict={ x: features_validation, y_: labels_validation, keep_prob: 1.0})
	if(validate_accuracy > 0.97):
    		checkpoint_file = os.path.join(FLAGS.train_dir, 'model.ckpt')
    		save_path = saver.save(sess, checkpoint_file)
    		print("Validate accuracy = {0}, Model saved in file: {1}".format(validate_accuracy,save_path))
    	
    if(i + 1) % FLAGS.printaccuracy_steps == 0:
    	train_accuracy = sess.run( accuracy, feed_dict={ x: features, y_: labels, keep_prob: 1.0})
    	validate_accuracy = sess.run( accuracy, feed_dict={ x: features_validation, y_: labels_validation, keep_prob: 1.0})
    	train_loss = sess.run( cross_entropy, feed_dict={ x: features, y_: labels, keep_prob: 1.0})
    	validation_loss = sess.run( cross_entropy, feed_dict={ x: features_validation, y_: labels_validation, keep_prob: 1.0})
    	print("step %d, train acc %g, validate acc %g, train loss %g, validate loss %g"%(i, train_accuracy, validate_accuracy, train_loss, validation_loss))

print("Training complete, took {0} minutes".format((time.time() - start_time)/60))

#print("test accuracy %g"% sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#filename = "/home/osboxes/Images/imgdata9.bin"
#a = np.fromfile(filename, dtype=np.float32)
#testimage = a.reshape(-1,784)

#res = sess.run(y_conv, feed_dict={x: testimage, keep_prob: 1.0})
#print(res)
#print(res.sum(axis=1))
#print(np.argmax(res))


import os
import scipy.io as sio
from utils import *
from cnn_class import cnn
from TfRnnAttention.attention import attention
import tensorflow as tf
import time
from tensorflow.contrib import rnn

tf.set_random_seed(33)
os.environ["CUDA_VISIBLE_DEVICES"]='0'

file_num= 1

num_node = 64

model = 'dg_cram'

data = sio.loadmat("./cross_subject_data_"+str(file_num)+".mat")

test_X	= data["test_x"]
train_X	= data["train_x"]

test_y	= data["test_y"].ravel()
train_y = data["train_y"].ravel()

train_y = np.asarray(pd.get_dummies(train_y.ravel()), dtype = np.int8)
test_y = np.asarray(pd.get_dummies(test_y.ravel()), dtype = np.int8)

if model == 'ng_cram':
	adj_type = 'ng'

elif model == 'dg_cram':
	adj_type = 'dg'

elif model == 'sg_cram':
	adj_type = 'sg'

adj = get_adj(num_node, adj_type)

test_X = np.matmul(np.expand_dims(adj, 0), test_X)
train_X = np.matmul(np.expand_dims(adj, 0), train_X)

window_size = 400
step = 10

train_raw_x = np.transpose(train_X, [0, 2, 1])
test_raw_x = np.transpose(test_X, [0, 2, 1])

train_win_x = segment_dataset(train_raw_x, window_size, step)
test_win_x = segment_dataset(test_raw_x, window_size, step)

# [trial, window, channel, time_length]
train_win_x = np.transpose(train_win_x, [0, 1, 3, 2])
test_win_x = np.transpose(test_win_x, [0, 1, 3, 2])

features_train = train_win_x
features_test = test_win_x
y_train = train_y
y_test = test_y

print("features_train shape:", features_train.shape)
print("features_test shape:", features_test.shape)

features_train = np.expand_dims(features_train, axis = -1)
features_test = np.expand_dims(features_test, axis = -1)

num_timestep = features_train.shape[1]
###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_height_1st	= 64
kernel_width_1st 	= 45

kernel_stride		= 1

conv_channel_num	= 40
# pooling parameter
pooling_height_1st 	= 1
pooling_width_1st 	= 75

pooling_stride_1st = 10
# full connected parameter
fc_size = 512
attention_size = 512
n_hidden_state = 64

n_fc_in = "None"

n_fc_out = "None"
###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# input height 
input_height = features_train.shape[2]

# input width
input_width = features_train.shape[3]

# prediction class
num_labels = 2
###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-5

# set maximum traing epochs
training_epochs = 110

# set batch size
batch_size = 10

# set dropout probability
dropout_prob = 0.5

# set train batch number per epoch
batch_num_per_epoch = features_train.shape[0]//batch_size

# instance cnn class
cnn_2d = cnn(padding='VALID')

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
train_phase = tf.placeholder(tf.bool, name = 'train_phase')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# first CNN layer
conv_1 = cnn_2d.apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride, train_phase)
pool_1 = cnn_2d.apply_max_pooling(conv_1, pooling_height_1st, pooling_width_1st, pooling_stride_1st)

pool1_shape = pool_1.get_shape().as_list()
pool1_flat = tf.reshape(pool_1, [-1, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])

fc_drop = tf.nn.dropout(pool1_flat, keep_prob)	

if (n_fc_in == 'None'):
	print("fc_in is None\n")
	lstm_in = tf.reshape(fc_drop, [-1, num_timestep, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])
else:
	lstm_in = tf.reshape(fc_drop, [-1, num_timestep, n_fc_in])

########################## RNN ########################
output = lstm_in
for layer in range(2):
    with tf.variable_scope('rnn_{}'.format(layer),reuse=False):
        cell_fw = tf.contrib.rnn.LSTMCell(n_hidden_state)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LSTMCell(n_hidden_state)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          output,
                                                          dtype=tf.float32)
        output = tf.concat(outputs,2)
        state = tf.concat(states,2)

rnn_op = output

########################## attention ########################
with tf.name_scope('Attention_layer'):
    attention_op, alphas = attention(rnn_op, attention_size, time_major = False, return_alphas=True)

attention_drop = tf.nn.dropout(attention_op, keep_prob)	
y_ = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2].value, num_labels)

# probability prediction 
y_posi = tf.nn.softmax(y_, name = "y_posi")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# set training SGD optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')


###########################################################################
# train test and save result
###########################################################################
# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
	true_test = []
	posi_test = []
	# training process
	for b in range(batch_num_per_epoch):
		offset = (b * batch_size) % (y_train.shape[0] - batch_size) 
		batch_x = features_train[offset:(offset + batch_size), :, :, :, :]
		batch_x = batch_x.reshape([len(batch_x)*num_timestep, num_node, window_size, 1])
		batch_y = y_train[offset:(offset + batch_size), :]
		_, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, train_phase: True})
	# calculate train and test accuracy after each training epoch
	if(epoch%1 == 0):
		train_accuracy 	= np.zeros(shape=[0], dtype=float)
		test_accuracy	= np.zeros(shape=[0], dtype=float)
		train_l 		= np.zeros(shape=[0], dtype=float)
		test_l			= np.zeros(shape=[0], dtype=float)
		# calculate train accuracy after each training epoch
		for i in range(batch_num_per_epoch):
			offset = (i * batch_size) % (y_train.shape[0] - batch_size) 
			train_batch_x = features_train[offset:(offset + batch_size), :, :, :]
			train_batch_x = train_batch_x.reshape([len(train_batch_x)*num_timestep, num_node, window_size, 1])
			train_batch_y = y_train[offset:(offset + batch_size), :]

			train_a, train_c = sess.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, train_phase: True})
			
			train_l = np.append(train_l, train_c)
			train_accuracy = np.append(train_accuracy, train_a)
		print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_l), "Training Accuracy: ", np.mean(train_accuracy))
		# calculate test accuracy after each training epoch
		for j in range(batch_num_per_epoch):
			offset = (j * batch_size) % (test_y.shape[0] - batch_size) 
			test_batch_x = features_test[offset:(offset + batch_size), :, :, :]
			test_batch_x = test_batch_x.reshape([len(test_batch_x)*num_timestep, num_node, window_size, 1])
			test_batch_y = y_test[offset:(offset + batch_size), :]
			
			test_a, test_c, test_p = sess.run([accuracy, cost, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, train_phase: True})
			
			test_accuracy = np.append(test_accuracy, test_a)
			test_l = np.append(test_l, test_c)
			true_test.append(test_batch_y)
			posi_test.append(test_p)

		auc_roc_test = roc_auc_score(y_true=np.array(true_test).reshape([-1, 2]), y_score = np.array(posi_test).reshape([-1, 2]))
		print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, "Test AUC: ", auc_roc_test, " Test Cost: ", np.mean(test_l), "Test Accuracy: ", np.mean(test_accuracy), "\n")

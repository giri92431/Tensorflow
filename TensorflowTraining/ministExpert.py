from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# x is placeholder vectore which is 2D matix which takes 784-dimision vectore of misist image 
#it is represented as 2-d tensor of floating point number 
x = tf.placeholder(tf.float32,[None,784])

"""
# w is the weight 
w has a shape of [784 ,10] because we want to multiply the
 784 dimision image vetors by [784,10] to procuce a 
 10 dimision vector of evidence for the difference classes 
"""
w = tf.Variable(tf.zeros([784,10]))

"""
b is the bias 
is [10] so we can add it to the output 

"""
b = tf.Variable(tf.zeros([10]))


# implement the model

# We start building the computation graph by creating nodes for the input images and target output classes.

#  first we multiply x by w with tf.matmul(x,w)  mathmultiply
# then add b to it and apply soft max of it 
y = tf.nn.softmax(tf.matmul(x,w) + b)


y_ = tf.placeholder(tf.float32, [None, 10])
# to implement cross entrophy first we need to first add a place holder t oinput the correct answer 


# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# # - sigma y' log(y)


# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# sess = tf.InteractiveSession()

# tf.global_variables_initializer().run()


# for _ in range(1000):
# 	batch_xs, batch_ys = mnist.train.next_batch(100)
# 	sess.run(train_step , feed_dict = {x:batch_xs, y_ : batch_ys })

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(accuracy,feed_dict={ x:mnist.test.images, y_: mnist.test.labels }))

def weight_variable (shape):
	initial = tf.truncated_normal(shape,stddev =0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial =tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2X2(X):
	return tf.nn.max_pool(X,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')



'''*** first convulation layer***'''
# First convolution 
# 32 feature for each 5X5 patch
# [5,5,1,5] first 2 are patch size , next tis the input channel , last is the output channel 
W_conv1 = weight_variable([5,5,1,32])

#  basise vector for componeent of output channel 
b_conv1 = bias_variable([32])


# we reshape the image to 4d vector 
x_image = tf.reshape(x,[-1,28,28,1])


# we then convel x_image with weighted tensor add the bias apply the relu function and finally max pool
# the  max_pool_2X2 will reduce the image to 14X14h_conv1 = tf.nn.relu(conv2d(x_image,W_Conv1))
h_pool1 = max_pool_2X2(h_conv1)



'''*** second convolation layer ***'''
# in ordeer ti build a deep network , we stack several layers of this type 
#  second layer will have 64 feature with 5X5 patch
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)


'''*** densely connected layer ***'''
# the image size has been reduce to 7X7, we add a fully-connected layer with 1024 neurons to allow processig on the entire image
#  reshape the tensor from the pooling layer into a batch of vectores 
# multiply by a weight matrix, add a bias and apply a relu
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


# droput 
'''
to reduce over fitting ,we will apply droup out before the readout layer 
create a place holder for the proberality the neurons output is keeping droped out 
this allows us to turn droup out on training and turn off on testing 
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.droupout(h_fc1,keep_prob)




# read out layer







































from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

#place holders 
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# varibles
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#initlizing globel variables
sess.run(tf.global_variables_initializer())


# implementing regression model 
y = tf.matmul(x,W) + b

# implementing loss function 
#loss function is the cross entrophy between target and the softmax activation function
# applied to th epredection model

# cross_entrophy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

# #optmize the loss 
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entrophy)

# Train the model
# using train_step
# for _ in range(1000):
# 	batch = mnist.train.next_batch(100)
# 	train_step.run(feed_dict={x:batch[0],y_:batch[1]})

# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Weight Initialization
# to create a model we are going to create a lot of weight and biase
# one should generally initlize weights with small amount of noise and symementry breaking and to prevent 0 GradientDescentOptimizer
# since usning ReLU as activation function 
# it is a good practice to initlize them with slightly positive inital bia to avoid dead neurons
def wieight_variable(shape):
	inital = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(inital)

def bias_variable(shape):
	inital = tf.constant(0.1,shape=shape)
	return tf.Variable(inital)

# Convolution and Pooling

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer
W_conv1 = wieight_variable([5, 5, 1, 32])
# The convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, 
# with the second and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.

x_image = tf.reshape(x, [-1, 28, 28, 1])

#***important**** The max_pool_2x2 method will reduce the image size to 14x14.
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second Convolutional Layer
# In order to build a deep network, we stack several layers of this type.
#  The second layer will have 64 features for each 5x5 patch
W_conv2 = wieight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
#***important**** now the image sizze will be reduced to 7x7
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
# we reshape the tensor from the polling layer into a batch of vectors, multply by weight matrix and adda bias and apply ReLU.
W_fc1 = wieight_variable( [7*7*64,1024] )
b_fc1 = bias_variable([1024])

h_pool2_flat =tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout: A Simple Way to Prevent Neural Networks fromOverfitting
# to reduce over fitting we apply drop outs  before read out layer 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout Layer
# Finally, we add a layer, just like for the one layer softmax regression above.
W_fc2 = wieight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
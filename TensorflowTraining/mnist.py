import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32,[None,784])
# x is placeholder vectore which is 2D matix which takes 784-dimision vectore of misist image 
#it is represented as 2-d tensor of floating point number 

w = tf.Variable(tf.zeros([784,10]))
"""
# w is the weight 
w has a shape of [784 ,10] because we want to multiply the
 784 dimision image vetors by [784,10] to procuce a 
 10 dimision vector of evidence for the difference classes 
"""
b = tf.Variable(tf.zeros([10]))
"""
b is the bias 
is [10] so we can add it to the output 

"""

# implement the model

y = tf.nn.softmax(tf.matmul(x,w) + b)
#  first we multiply x by w with tf.matmul(x,w)  mathmultiply
# then add b to it and apply soft max of it 


y_ = tf.placeholder(tf.float32, [None, 10])
# to implement cross entrophy first we need to first add a place holder t oinput the correct answer 


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# - sigma y' log(y)


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step , feed_dict = {x:batch_xs, y_ : batch_ys })

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy,feed_dict={ x:mnist.test.images, y_: mnist.test.labels }))



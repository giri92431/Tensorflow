import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# create a computational graph 

#Constants
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
# print(node1, node2)


# running a computational graph 

sess = tf.Session()
# print(sess.run([node2,node1]))

# PlaceHolder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+ b # short cut to tf.add(a,b)

# print(sess.run(adder_node,{a:2,b:4.2}))

# print (sess.run())
# print(sess.run(adder_node,{a:[13,14] , b:[ 1 , 34.2]}))


add_and_tripple = adder_node *3 

# print(sess.run(add_and_tripple,{a:1,b:12}))


# variables 

w = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x= tf.placeholder(tf.float32)

linear_model = w * x + b

init = tf.global_variables_initializer()



# print(sess.run(linear_model,{x:[1,2,3,4]}))



y = tf.placeholder(tf.float32)

squared_delta = tf.square(linear_model - y )

loss = tf.reduce_sum(squared_delta)

# print(sess.run(squared_delta,{x:[1,2,3,4], y:[0,-1,-2,-3]}))

# print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))

# guss the value of W and b to be -1 ,1 
"""
fixW = tf.assign(w,[-1.])
fixb= tf.assign(b , [1.])

sess.run([fixW,fixb])

print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))
# the loss value will become 0
"""
#tf.train api

x_train = [1,2,3,4,5]
y_train = [0,-1,-2,-3,-4] 

Optimizer = tf.train.GradientDescentOptimizer(0.01)
train = Optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	sess.run(train,{x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


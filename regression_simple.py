#import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU
#tf.compat.v1.disable_eager_execution() ##this s solving the graph problem in Session
"""
n_features = 10
n_dense_neurons = 3
x = tf.placeholder(tf.float32,(None,n_features))
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x,W)
z = tf.add(xW,b)
a = tf.sigmoid(z)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})
print(layer_out) #sigmoid kullandığımız için neuronlar 0 ve 1 arasında
"""
#####SIMPLE REGRESSION EXAMPLE########
import matplotlib.pyplot as plt
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
plt.plot(x_data,y_label,'*')
#plt.show()
#y = mx + b
#np.random.rand(2) #0.44, 0.87
m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x + b
    #hata için cost function lazım
    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])
x_test = np.linspace(-1,11,10)
#y = mx+b
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')
#plt.plot(x_data,y_label,'*')
plt.show()
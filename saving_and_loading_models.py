import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

#y = mx + b

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
plt.plot(x_data,y_label,'*')
plt.show()
np.random.rand(2)
m = tf.Variable(0.40)
b = tf.Variable(0.3)

error = tf.reduce_mean(y_label - (m*x_data+b))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    for i in range(epochs):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])
    saver.save(sess,'models/my_first_model.ckpt')

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')
plt.show()

with tf.Session() as sess:
    saver.restore(sess,'models/my_first_model.ckpt')
    restored_slope, restored_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
y_pred_plot = restored_slope*x_test + restored_intercept
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')
plt.title('Restored')
plt.show()



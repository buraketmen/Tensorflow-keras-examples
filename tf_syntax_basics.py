import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU
#tf.compat.v1.disable_eager_execution() ##this s solving the graph problem in Session
#######################################################
"""
tf.compat.v1.disable_eager_execution() ##this s solving the graph problem in Session
a = tf.constant('Hello ')
b = tf.constant('World')
sess = tf.compat.v1.Session()
print(sess.run(a+b))
"""
#######################################################
"""
tf.compat.v1.disable_eager_execution() #this s solving the graph problem in Session
const = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random.normal((4,4),mean=0,stddev=1.0)
myrandu = tf.random.uniform((4,4),minval=0,maxval=1)

my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]
sess = tf.compat.v1.Session()
for op in my_ops:
    #print(op.eval(session=sess))
    print(sess.run(op))
    print('\n')
"""
#######################################################
"""
tf.compat.v1.disable_eager_execution() #this s solving the graph problem in Session
a = tf.constant([[1,2],[3,4]])
print(a.get_shape())
b = tf.constant([[10],[100]])
print(b.get_shape())
result = tf.matmul(a,b)
sess = tf.compat.v1.Session()
print(sess.run(result))
print(result.eval(session=sess))
"""
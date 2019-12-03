import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU

x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
# y = mx +b and b = 5
y_true = (0.5 * x_data) + 5 +noise

feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train,x_eval,y_train,y_eval =  train_test_split(x_data,y_true,test_size=0.3,random_state=101)
#print(x_train.shape)
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)
estimator.train(input_fn=input_func,steps=1000)
train_metrics = estimator.evaluate(input_fn = train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)
#Training data metrics
print(train_metrics)
#Eval metrics
print(eval_metrics)

brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)
#print(list(estimator.predict(input_fn=input_fn_predict)))
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])
print(predictions)

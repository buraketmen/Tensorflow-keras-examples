import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import osk

from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU

diabetes = pd.read_csv('Data/pima-indians-diabetes.csv')
#print(diabetes.columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration',
                'Blood_pressure', 'Triceps','Insulin', 'BMI',
                'Pedigree']
#normalizasyon
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply\
    (lambda x:(x-x.min())/(x.max()-x.min()))

#numeric columns
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#object columns with vocabulary
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
#object columns with hash, bucket size is the max value of category
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)
diabetes['Age'].hist(bins=20)
plt.title('Age')
plt.show()
#continuous numeric to categorical column
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]

#train and test split
x_data = diabetes.drop('Class',axis=1) #only features
labels = diabetes['Class'] #only label

X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.3,random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                 batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,
                                                      batch_size=10,num_epochs=1,shuffle=False)
results = model.evaluate(eval_input_func)
print(results)
#use this for new data, but we dont have. thats why we re using Xtest
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,
                                                      num_epochs=1,shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
print(my_pred)
#NN

#dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)
#dnn_model.train(input_fn=input_func,steps=1000) Error because of group
embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,embedded_group_col,age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,
                                                 shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
results = dnn_model.evaluate(eval_input_func)
print(results)

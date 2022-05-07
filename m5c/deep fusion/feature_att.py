from keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Bidirectional
from keras.layers import LSTM
#from keras.layers import Attention
from attention import Attention

model = Sequential()
model.add(Attention())
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

data_=pd.read_csv(r"input file",header=None)
data=np.array(data_)
data=data[:,0:]
[m1,n1]=np.shape(data)
shu=scale(data)
X1=shu

X=np.reshape(X1,(-1,1,n1))
cv_clf = model
tf.config.experimental_run_functions_eagerly(True)
feature=cv_clf.predict(X)
data_csv = pd.DataFrame(data=feature)
data_csv.to_csv(r'save file',header=None,index=False)

from re import X
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, fbeta_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from tensorflow.python.keras.backend import shape
from tensorflow.python.ops.gen_array_ops import Reshape
import utils.tools as utils
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Add, Dense, Activation, GRU, BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, Concatenate, Reshape
from keras.models import Model, load_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def CG(x):
    x1 = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(32, kernel_size=1, strides=1, padding='same',activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32,activation='relu')(x1)
    x2 = GRU(32)(x)
    x = Add()([x1,x2])
    x = Activation('relu')(x)
    x = Reshape((32,1))(x)
    return x


input = Input(shape=(256,1))
x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(input)
x = MaxPooling1D()(x)
x = CG(x)
x = CG(x)
x = CG(x)
x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
x = MaxPooling1D()(x)
x = Flatten()(x)

x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(2, activation='sigmoid')(x)
model = Model(inputs = input, outputs = x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

data_=pd.read_csv(r'input file',header=None)
data=np.array(data_)
data=data[:,0:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
#label1=np.ones((544,1))#Value can be changed
#label2=np.zeros((407,1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label
sepscores = []

ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

X = X.reshape(m1, n1,1)

#model.fit([X,X], [y,y], epochs=10)

skf= StratifiedKFold(n_splits=7)

for train, test in skf.split(X,y):
    y_train=utils.to_categorical(y[train])#generate the resonable results
    cv_clf = model
    hist=cv_clf.fit(X[train],
                    y_train,
                    epochs=5)
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]
    y_score=cv_clf.predict(X[test])#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    #hist=[]
    #cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
#yscore_sum.to_csv('yscore_imCNN_SBP_test.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
#ytest_sum.to_csv('ytest_imCNN_SBP_test.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='DL_1 ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv(r'save file')

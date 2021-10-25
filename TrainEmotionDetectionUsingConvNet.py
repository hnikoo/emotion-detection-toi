import numpy as np
import sys
import os
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Input, Reshape
from keras.models import Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D,UpSampling1D,ZeroPadding1D
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from time import strftime

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,confusion_matrix
from sklearn.metrics import roc_auc_score,auc,f1_score,precision_score,recall_score,accuracy_score
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV
import fnmatch
import cPickle

from utils.load_data import Balance_Dataset,load_cpickle,load_cpickle_withCLASSES,load_cpickle_withLabel
from utils.load_data import loadData_LikeDislike,GET_LAST_LAYER_REP_LIKEDISLIKE


tstring = strftime("%Y_%m_%d_%H_%M")
pid = os.getpid()

tstring = str(pid) + '_' + tstring

sys.setrecursionlimit(10000)

roi_n = 9#*5
LenSig = 448
z_mean,z_log_var = 0,0
n_label = 7

BANDS = ['data']#,'Mayer','thermal','TH','HeartBand']

class0 =[1,2,6] # suprise,joy,tranq
class1 = [5,3,4]# sad,disgust,bored
ClassesNames = '_SevenEmotions_'

#class0 =[4,6] # bored,tranq
#class1 = [0,1]# fear,suprise
#ClassesNames = '_AROUSAL_Augment_'



def DefineModel(Version=1):
    if Version == 1:
        model = Sequential()
        model.add(Convolution1D(32, 15, border_mode='same', input_shape=(LenSig, roi_n)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_length=3))
        model.add(Dropout(0.5))
        model.add(Flatten()) 
    
        model.add(Dense(2024))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
    
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(n_label))
        model.add(Activation('softmax'))  
    if Version == 2:
        # encoder
        input_signal = Input(shape=(LenSig, roi_n))
        x = Convolution1D(128, 15, activation='relu',border_mode='same')(input_signal)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_length=3)(x)
        x = Dropout(0.5)(x)
        x = Convolution1D(128, 10, activation='relu', border_mode='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_length=3)(x)
        x = Convolution1D(64, 6, activation='relu',border_mode='same')(x)
        x = MaxPooling1D(pool_length=3)(x)
        x = Flatten()(x)
        x = Dense(128)(x) 
        x = Activation('relu')(x)
        x = Dense(n_label)(x)
        z = Activation('softmax')(x)
        model = Model(input_signal,z)
    if Version == 3:
        model = Sequential()
        model.add(Convolution1D(32, 190, border_mode='valid', input_shape=(LenSig, roi_n)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_length=3))
        model.add(Dropout(0.5))
        model.add(Flatten()) 
    
        model.add(Dense(2024))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
    
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(n_label))
        model.add(Activation('softmax'))    
        

    return model

   

def TrainDeepNetwork(model,X_train,y_train,X_test,y_test,PreLoadModel=None): 
    
    if PreLoadModel is not None:
        VAE,encoder = DefineVAEModel(Version=1)
        VAE.load_weights(PreLoadModel)
        W = VAE.get_weights()
        W2 = []
        for i in xrange(0,8):
            W2.append(W[i])
        W_old = model.get_weights()
        W2.append(W_old[8])
        W2.append(W_old[9])
        model.set_weights(W2)

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])


    mcp = ModelCheckpoint(filepath="best_model"+ClassesNames+tstring+".hdf5", 
                          verbose=1,
                          monitor='val_loss',save_best_only=True)


    history = model.fit(X_train, y_train,
                        batch_size=300, 
                        nb_epoch=150,
                        verbose=1, callbacks=[mcp],
                        validation_data=(X_test,y_test))

   

def TestModel(model,X_train,y_train,X_test,y_test,Filename='./best_model_2016_07_19_11_54.hdf5',verbose=True):
    model.load_weights(Filename)
    y = model.predict(X_test, batch_size=X_test.shape[0])
    y = np.argmax(y,axis=1)
    k = np.argmax(y_test,axis=1)
    if verbose:
        print 'accuracy: ', accuracy_score(k,y)
        print 'precision: ', precision_score(k,y)
        print 'recall: ' , recall_score(k,y)
        print 'f1 score: ', f1_score(k,y)
        print confusion_matrix(k,y)
        M = confusion_matrix(k,y)
        print 'Precision Per Class: ', np.diag(M).astype(float)/np.sum(M,axis=0) 
        print 'Recall Per Class: ', np.diag(M).astype(float)/np.sum(M,axis=1)

    #return accuracy_score(k,y),precision_score(k,y),recall_score(k,y),f1_score(k,y),confusion_matrix(k,y)
    return accuracy_score(k,y),confusion_matrix(k,y)
    
  
def loadCNNEncoder(memodelFile):
    model = Sequential()
    model.add(Convolution1D(32, 15, border_mode='same', input_shape=(LenSig, roi_n)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(0.5))
    model.add(Flatten()) 

    model.add(Dense(2024))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(128))
    #model.add(Activation('relu'))    
    
    wm = DefineModel()
    wm.load_weights(memodelFile)
    W = wm.get_weights()
    newW = []
    for i in xrange(6):
        newW.append(W[i])
        
    model.set_weights(newW)
    return model
        


def PlotFilters(model,FILE):
    model.load_weights(FILE)
    W = model.get_weights()
    FirstLayer = W[0]
    plt.figure(1)
    
    for i in xrange(1,17):
        plt.subplot(4,4,i)
        plt.plot(FirstLayer[:,0,0,i])
    plt.title('Different Filters')
    
    plt.figure(2)
    for i in xrange(1,9):
            plt.subplot(4,4,i)
            plt.plot(FirstLayer[:,0,i,0])    
    plt.title('Filter of ROIs')
    
    mat = {'W':FirstLayer}
    savemat('FirstLayer.mat',mat)
    
    plt.show()


def showClusters(model,FILE,X_test,y_test):
    encoder = loadCNNEncoder(FILE)
    new_X = encoder.predict(X_test)
    clf = TSNE(perplexity=10.0)
    xx = clf.fit_transform(new_X)
    clsNames = ['Fear','Sup','Joy','Disgust','Bored','Sad','Tranq','Cloud']
    colors=['b', 'c', 'y', 'm', 'r','k','g','#eeefff']
    plt.figure(1)
    for c in [0,1,2,3,4,5,6,7]:
        idx = np.where(np.argmax(y_test,axis=1)==int(c))[0]
        plt.scatter(xx[idx,0],xx[idx,1],s=120,color=colors[c],label=clsNames[c])
    plt.legend(loc=4)
    plt.title('TSNE 2D projection')
    plt.show()


def main():
    global roi_n,LenSig,n_label,BANDS,class0,class1,ClassesNames
    roi_n = 9
    LenSig = 448
    n_label = 6 
    BANDS = ['data_f2']#,'Mayer','thermal','TH','HeartBand']  HeartBand: 0.7-2Hz raw: 0-30Hz  data_f: 0.2-2Hz , data_f2: 0-2Hz
    #class0 =[2,6] # joy,tranq
    #class1 = [0,5,3,4,1]# Fear,sad,disgust,bored,suprise
    ClassesNames = '_data_f2_6class_'    

    Tra_OR_Test = 'Train'
    fileid = ''   
    
    if Tra_OR_Test == 'Train':
        X_train,y_train,X_val,y_val,X_test,y_test = load_cpickle_withLabel(
            File=None,
            datadir='/data_shared/DATA/Emovid600/bloodflow_bands/',
            markerdir='/data_shared/DATA/Emovid600/rateMarker/',
            LABELS = [0,2,3,4,5,6],
            roi_n=roi_n, 
            LenSig=LenSig, 
            n_label=n_label,
            BANDS=BANDS,
            ClassesNames=ClassesNames,
            tstring=tstring
        )
        
        X_train,y_train = Balance_Dataset(X_train,y_train);
        X_val,y_val = Balance_Dataset(X_val,y_val);
        
        model = DefineModel(Version=1)
        L =X_train.shape[0]
        idx = np.random.permutation(X_train.shape[0])
        TrainDeepNetwork(model,X_train[idx,],y_train[idx,],X_val,y_val)
   
        
    if Tra_OR_Test == 'Test':
        
        X_train,y_train,X_val,y_val,X_test,y_test = load_cpickle_withLabel(
            File='TrainData'+ClassesNames+fileid+'.pickle',
            datadir='/data_shared/DATA/Emovid600/bloodflow_bands/',
            markerdir='/data_shared/DATA/Emovid600/rateMarker/',
            LABELS = [0,2,3,4,5,6],
            roi_n=roi_n, 
            LenSig=LenSig, 
            n_label=n_label,
            BANDS = BANDS
            )     
        #X_train,y_train,X_val,y_val,X_test,y_test = load_cpickle_withCLASSES(
        #           File='../emovidVAE/newFeature64.pickle',CLASSES=[class0,class1]) 
        
        X_test,y_test = Balance_Dataset(X_test,y_test);
        model = DefineModel(Version=1)
        TestModel(model,X_train,y_train,X_test,y_test,Filename='./best_model'+ClassesNames + fileid+ '.hdf5')
        
 
        
        
if __name__ == "__main__":
    main()
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

   

def test_on_files():
    dirname = './data_f2_0_2Hz/'
    global roi_n,LenSig,n_label,BANDS,class0,class1,ClassesNames
    roi_n = 9
    LenSig = 448
    n_label = 6 
    BANDS = ['data_f2']#,'Mayer','thermal','TH','HeartBand']    
    #class0 =[2,6] # joy,tranq
    #class1 = [0,5,3,4,1]# Fear,sad,disgust,bored,suprise
    ClassesNames = '_data_f2_6class_'   
    
    
    files = os.listdir(dirname)
    hdfFiles = []
    for afile in files:
        if fnmatch.fnmatch(afile,'best_model'+ClassesNames+'*.hdf5'):
            hdfFiles.append(afile)
    results =[]
    for hdf5file in hdfFiles:
        dataName = dirname+'TrainData'+ClassesNames+ hdf5file[26:-5] + '.pickle'
        X_train,y_train,X_val,y_val,X_test,y_test = load_cpickle_withLabel(
            File=dataName,
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
        model = DefineModel()
        #acc,pr,rec,f1score,conf = TestModel(model,X_train,y_train,X_val,y_val,Filename=dirname+hdf5file,verbose=False)
        acc,conf = TestModel(model,X_train,y_train,X_val,y_val,Filename=dirname+hdf5file,verbose=False)
        print hdf5file
        print acc
        #results.append([acc,pr,rec,f1score,conf])
        results.append([acc,0,0,0,conf])
        
    with open('results_data_f_6class.pickle','w') as f:
        cPickle.dump(results,f)
        
        
        
if __name__ == "__main__":
    test_on_files()
import numpy as np
from scipy.io import savemat,loadmat
import cPickle
import os
from scipy.stats import zscore
from scipy.signal import decimate
from sklearn.cross_validation import train_test_split



def Balance_Dataset(X,y):
    n_example = y.shape[0]
    n_label = y.shape[1]
    nExPerClass = []
    for i in xrange(n_label):
        nExPerClass.append(np.int(y[:,i].sum()))
    
    MIN_NUM_Ex = nExPerClass[np.argmin(np.asarray(nExPerClass))]
    X_tmp = []
    y_tmp = []
    for i,num in enumerate(nExPerClass):
        idx = np.random.permutation(num)
        clidx = np.where(y[:,i]>0)[0]
        clidx = clidx[idx[:MIN_NUM_Ex]]
        X_tmp.append(X[clidx,])
        y_tmp.append(y[clidx,])
        
    X = np.concatenate(X_tmp,axis=0)
    y = np.concatenate(y_tmp,axis=0)
    
        
    return X,y
        
        
def load_cpickle_withCLASSES(File=None,
                            datadir = '../bloodflow_bands/',
                            markerdir = '../rateMarker/',
                            CLASSES =None,
                            roi_n = 9 ,
                            LenSig = 448,
                            n_label = 8,
                            BANDS = ['data'],
                            ClassesNames=None,
                            tstring=None):
    
    if CLASSES is None:
        print 'Specify Classes! ...'
        return None
    
    if File is None:
        marker_list = []
        data_list = []
    
        for item in os.listdir(datadir):
            data_list.append(item)
            marker_list.append(item)
    
        idxs = np.arange(len(data_list))
        idxs_train, idxs_test = train_test_split(idxs, test_size=0.1)
        idxs = np.arange(idxs_train.shape[0])
        idxs_train2, idxs_test2 = train_test_split(idxs, test_size=0.1)
        idxs_validation = idxs_train[idxs_test2]
        idxs_train = idxs_train[idxs_train2]  
        mat = {'data_list':data_list,'marker_list':marker_list,'idxs_train':idxs_train,
               'idxs_test':idxs_test,'idxs_validation':idxs_validation}
        with open('TrainData'+ClassesNames+tstring+'.pickle','w') as f:
            cPickle.dump(mat,f)
    else:
        with open(File,'r') as f:
            mat = cPickle.load(f)
        data_list = list(mat['data_list'])
        marker_list = list(mat['marker_list'])
        idxs_train = np.asarray(mat['idxs_train'])
        idxs_test = np.asarray(mat['idxs_test'])
        idxs_validation = np.asarray(mat['idxs_validation'])    
   
        
    data_mat_list = []
    label_mat_list = []

    for data_file,marker_file in  zip(data_list,marker_list):
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat, label_mat = load_mat(data,
                                       marker, 
                                       roi_n=roi_n, 
                                       LenSig=LenSig,
                                       n_label=n_label)     
        data_mat_list.append(data_mat)
        label_mat_list.append(label_mat) 
        
    data_mat_t = np.concatenate(data_mat_list, axis=0)
    label_mat_t = np.concatenate(label_mat_list, axis=0)
    
    wherenan = np.isnan(data_mat_t)
    whereinf = np.isinf(data_mat_t)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat_t[nanidx] = 0
    data_mat_t[infidx] = 0

    train_data = data_mat_t[idxs_train, ]
    test_data = data_mat_t[idxs_test, ]
    train_label = label_mat_t[idxs_train, ]
    test_label = label_mat_t[idxs_test, ]
    val_data = data_mat_t[idxs_validation, ]
    val_label = label_mat_t[idxs_validation, ]    
    train_data_flat = np.reshape(train_data, (-1, LenSig, roi_n))
    train_label_flat = np.reshape(train_label, (-1))
    val_data_flat = np.reshape(val_data, (-1, LenSig, roi_n))
    val_label_flat = np.reshape(val_label, (-1))    
    test_data_flat = np.reshape(test_data, (-1, LenSig, roi_n))
    test_label_flat = np.reshape(test_label, (-1))


    X_train = train_data_flat[:]
    y_train = train_label_flat[:]
    X_val = val_data_flat[:]
    y_val = val_label_flat[:]
    X_test = test_data_flat[:]
    y_test = test_label_flat[:]
    
    def find_class(CLASSES,i):
        for index,item in enumerate(CLASSES):
            if i in item:
                return index
            
    ITEMLIST = [item for sublist in CLASSES for item in sublist]
    idx = np.where(np.in1d(y_train, ITEMLIST)  )[0]
    y_train = y_train[idx]
    X_train = X_train[idx]
    y_tmp = np.zeros((y_train.shape[0],len(CLASSES)))
    for ix,i in enumerate(y_train):
        y_tmp[ix,find_class(CLASSES,i)] = 1
    y_train = np.copy(y_tmp)
    
    
    idx = np.where(np.in1d(y_test, ITEMLIST) )[0]
    y_test = y_test[idx]
    X_test = X_test[idx]
    y_tmp = np.zeros((y_test.shape[0],len(CLASSES)))
    for ix,i in enumerate(y_test):
        y_tmp[ix,find_class(CLASSES,i)] = 1
    y_test = np.copy(y_tmp) 
    
    
    idx = np.where(np.in1d(y_val, ITEMLIST) )[0]
    y_val = y_val[idx]
    X_val = X_val[idx]
    y_tmp = np.zeros((y_val.shape[0],len(CLASSES)))
    for ix,i in enumerate(y_val):
        y_tmp[ix,find_class(CLASSES,i)] = 1
    y_val = np.copy(y_tmp)    
    
    return X_train,y_train,X_val,y_val,X_test,y_test

def load_cpickle_withLabel(File=None,
                           datadir = '../bloodflow_bands/',
                           markerdir = '../rateMarker/',
                           LABELS = [0,1,2,3,4,5,6],
                           roi_n = 9 ,
                           LenSig = 448,
                           n_label = 8,
                           BANDS = ['data'],
                           ClassesNames=None,
                           tstring=None):
    
    if File is None:
        marker_list = []
        data_list = []
    
        for item in os.listdir(datadir):
            data_list.append(item)
            marker_list.append(item)
    
        idxs = np.arange(len(data_list))
        idxs_train, idxs_test = train_test_split(idxs, test_size=0.1)
        idxs = np.arange(idxs_train.shape[0])
        idxs_train2, idxs_test2 = train_test_split(idxs, test_size=0.1)
        idxs_validation = idxs_train[idxs_test2]
        idxs_train = idxs_train[idxs_train2]  
        mat = {'data_list':data_list,'marker_list':marker_list,'idxs_train':idxs_train,
               'idxs_test':idxs_test,'idxs_validation':idxs_validation}
        with open('TrainData'+ClassesNames+tstring+'.pickle','w') as f:
            cPickle.dump(mat,f)
    else:
        with open(File,'r') as f:
            mat = cPickle.load(f)
        data_list = list(mat['data_list'])
        marker_list = list(mat['marker_list'])
        idxs_train = np.asarray(mat['idxs_train'])
        idxs_test = np.asarray(mat['idxs_test'])
        idxs_validation = np.asarray(mat['idxs_validation'])  

        
    data_mat_list = []
    label_mat_list = []

    for data_file,marker_file in  zip(data_list,marker_list):
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat, label_mat = load_mat(data,
                                       marker, 
                                       roi_n=roi_n, 
                                       LenSig=LenSig,
                                       n_label=n_label,
                                       BANDS = BANDS)   
        data_mat_list.append(data_mat)
        label_mat_list.append(label_mat) 
        
    data_mat_t = np.concatenate(data_mat_list, axis=0)
    label_mat_t = np.concatenate(label_mat_list, axis=0)
    
    wherenan = np.isnan(data_mat_t)
    whereinf = np.isinf(data_mat_t)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat_t[nanidx] = 0
    data_mat_t[infidx] = 0

    train_data = data_mat_t[idxs_train, ]
    test_data = data_mat_t[idxs_test, ]
    train_label = label_mat_t[idxs_train, ]
    test_label = label_mat_t[idxs_test, ]
    val_data = data_mat_t[idxs_validation, ]
    val_label = label_mat_t[idxs_validation, ]    
    train_data_flat = np.reshape(train_data, (-1, LenSig, roi_n))
    train_label_flat = np.reshape(train_label, (-1))
    val_data_flat = np.reshape(val_data, (-1, LenSig, roi_n))
    val_label_flat = np.reshape(val_label, (-1))    
    test_data_flat = np.reshape(test_data, (-1, LenSig, roi_n))
    test_label_flat = np.reshape(test_label, (-1))


    X_train = train_data_flat[:]
    y_train = train_label_flat[:]
    X_val = val_data_flat[:]
    y_val = val_label_flat[:]
    X_test = test_data_flat[:]
    y_test = test_label_flat[:]
    
    idx = np.where(np.in1d(y_train, LABELS)  )[0]
    y_train = y_train[idx]
    X_train = X_train[idx]
    y_tmp = np.zeros((y_train.shape[0],len(LABELS)))
    for ix,i in enumerate(y_train):
        y_tmp[ix,LABELS.index(i)] = 1
    y_train = np.copy(y_tmp)
    
    
    idx = np.where(np.in1d(y_test, LABELS) )[0]
    y_test = y_test[idx]
    X_test = X_test[idx]
    y_tmp = np.zeros((y_test.shape[0],len(LABELS)))
    for ix,i in enumerate(y_test):
        y_tmp[ix,LABELS.index(i)] = 1
    y_test = np.copy(y_tmp) 
    
    
    idx = np.where(np.in1d(y_val, LABELS) )[0]
    y_val = y_val[idx]
    X_val = X_val[idx]
    y_tmp = np.zeros((y_val.shape[0],len(LABELS)))
    for ix,i in enumerate(y_val):
        y_tmp[ix,LABELS.index(i)] = 1
    y_val = np.copy(y_tmp)    
    
    return X_train,y_train,X_val,y_val,X_test,y_test
    
    
def load_cpickle(File=None,
                 datadir = '../bloodflow_bands/',
                 markerdir = '../rateMarker/',
                 class0=None,
                 class1=None,
                 roi_n = 9 ,
                 LenSig = 448,
                 n_label = 8,
                 BANDS = ['data'],
                 ClassesNames=None,
                 tstring=None):

    if class0 is None or class1 is None:
        print 'Specify class0 and class1! ...';
        return None
    
    if File is None:
        marker_list = []
        data_list = []
    
        for item in os.listdir(datadir):
            data_list.append(item)
            marker_list.append('EmoVid'+item[6:])
    
        idxs = np.arange(len(data_list))
        idxs_train, idxs_test = train_test_split(idxs, test_size=0.1)
        idxs = np.arange(idxs_train.shape[0])
        idxs_train2, idxs_test2 = train_test_split(idxs, test_size=0.1)
        idxs_validation = idxs_train[idxs_test2]
        idxs_train = idxs_train[idxs_train2]  
        mat = {'data_list':data_list,'marker_list':marker_list,'idxs_train':idxs_train,
               'idxs_test':idxs_test,'idxs_validation':idxs_validation}
        with open('TrainData'+ClassesNames+tstring+'.pickle','w') as f:
            cPickle.dump(mat,f)
    else:
        with open(File,'r') as f:
            mat = cPickle.load(f)
        data_list = list(mat['data_list'])
        marker_list = list(mat['marker_list'])
        idxs_train = np.asarray(mat['idxs_train'])
        idxs_test = np.asarray(mat['idxs_test'])
        idxs_validation = np.asarray(mat['idxs_validation'])   
        
    data_mat_list = []
    label_mat_list = []

    for data_file,marker_file in  zip(data_list,marker_list):
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat, label_mat = load_mat(data,
                                       marker, 
                                       roi_n=roi_n, 
                                       LenSig=LenSig,
                                       n_label=n_label,
                                       BANDS=BANDS)   
        data_mat_list.append(data_mat)
        label_mat_list.append(label_mat) 
        
    data_mat_t = np.concatenate(data_mat_list, axis=0)
    label_mat_t = np.concatenate(label_mat_list, axis=0)
    
    wherenan = np.isnan(data_mat_t)
    whereinf = np.isinf(data_mat_t)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat_t[nanidx] = 0
    data_mat_t[infidx] = 0

    train_data = data_mat_t[idxs_train, ]
    test_data = data_mat_t[idxs_test, ]
    train_label = label_mat_t[idxs_train, ]
    test_label = label_mat_t[idxs_test, ]
    val_data = data_mat_t[idxs_validation, ]
    val_label = label_mat_t[idxs_validation, ]    
    train_data_flat = np.reshape(train_data, (-1, LenSig, roi_n))
    train_label_flat = np.reshape(train_label, (-1))
    val_data_flat = np.reshape(val_data, (-1, LenSig, roi_n))
    val_label_flat = np.reshape(val_label, (-1))    
    test_data_flat = np.reshape(test_data, (-1, LenSig, roi_n))
    test_label_flat = np.reshape(test_label, (-1))


    X_train = train_data_flat[:]
    y_train = train_label_flat[:]
    X_val = val_data_flat[:]
    y_val = val_label_flat[:]
    X_test = test_data_flat[:]
    y_test = test_label_flat[:]
    
    idx1 = np.where(np.in1d(y_train, class0)  )[0]
    idx2 = np.where(np.in1d(y_train, class1) )[0]
    y_train[idx1]=0
    y_train[idx2]=1
    idx = np.concatenate((idx1,idx2))
    y_train = y_train[idx]
    X_train = X_train[idx]
    y_train = np.asarray((1-y_train,y_train)).T    
    
    idx1 = np.where(np.in1d(y_test, class0) )[0]
    idx2 = np.where(np.in1d(y_test, class1) )[0]
    y_test[idx1]=0
    y_test[idx2]=1
    idx = np.concatenate((idx1,idx2))
    y_test = y_test[idx]
    X_test = X_test[idx]
    y_test = np.asarray((1-y_test,y_test)).T 
    
    
    idx1 = np.where(np.in1d(y_val, class0) )[0]
    idx2 = np.where(np.in1d(y_val, class1) )[0]
    y_val[idx1]=0
    y_val[idx2]=1
    idx = np.concatenate((idx1,idx2))
    y_val = y_val[idx]
    X_val = X_val[idx]
    y_val = np.asarray((1-y_val,y_val)).T    
    
    return X_train,y_train,X_val,y_val,X_test,y_test
    
def load_mat(data,
             marker,
             nshifts=5,
             windowsize=15,
             backward=False,
             roi_n = 9 ,
             LenSig = 448,
             n_label = 7,
             BANDS=None):
    
    if backward:
        num_trials = (marker.shape[0] - 3)*2*nshifts
        shiftbegin = -nshifts
    else:
        num_trials = (marker.shape[0] - 3)*nshifts
        shiftbegin = 0
        
    nBands = len(BANDS)
    BandROINUM = {'data':9,'HeartBand':9,'Mayer':9,'TH':9,'thermal':9,'HR':1,'SD1':1,'SD2':1,'data_f':9,'data_f2':9}
    Fraction = 4.0; # downsample from 60Hz to 15Hz
        
    data_mat = np.zeros((1, num_trials, LenSig, roi_n))
    label_mat = np.zeros((1, num_trials))
    dwsampleData = {}
    for item in BANDS:
        dwsampleData[item] = np.zeros((np.ceil(data[item].shape[1]/Fraction),BandROINUM[item]))
        for q in xrange(BandROINUM[item]):
            dwsampleData[item][:,q]= decimate(data[item][q,:], int(Fraction)) 
 
    index = 0
    for i, r in enumerate(marker[2:30,]):
        trigger = r[1]
        position = r[0]
        
        for shifts in xrange(shiftbegin,nshifts):
            position = position + shifts*windowsize
            ind = 0
            for item in BANDS :
                start = np.floor(position/Fraction) 
                end = np.floor(position/Fraction) + LenSig
                if end > dwsampleData[item].shape[0]:
                    continue;
                chopSignals = dwsampleData[item][start:end,:]               

                data_mat[0, index, :,ind:ind+BandROINUM[item]] = zscore(chopSignals,axis=0)
                ind += BandROINUM[item]
                
            label_mat[0, index] = trigger
            index += 1
               
            
    return data_mat, label_mat


def load_mat_likeDislike(data,
                         marker,
                         nshifts=5,
                         windowsize=15,
                         backward=False,
                         roi_n = 9 ,
                         LenSig = 448,
                         n_label = 7,
                         BANDS=None):
    
    if backward:
        num_trials = (marker.shape[0] - 4)*2*nshifts
        shiftbegin = -nshifts
    else:
        num_trials = (marker.shape[0] - 4)*nshifts
        shiftbegin = 0
        
    nBands = len(BANDS)
    BandROINUM = {'data':9,'HeartBand':9,'Mayer':9,'TH':9,'thermal':9,'HR':1,'SD1':1,'SD2':1}
    Fraction = 4.0; # downsample from 60Hz to 15Hz
        
    data_mat = np.zeros((1, num_trials, LenSig, roi_n))
    label_mat = np.zeros((1, num_trials))
    dwsampleData = {}
    for item in BANDS:
        dwsampleData[item] = np.zeros((np.ceil(data[item].shape[1]/Fraction),BandROINUM[item]))
        for q in xrange(BandROINUM[item]):
            dwsampleData[item][:,q]= decimate(data[item][q,:], int(Fraction)) 
 
    index = 0
    for i, r in enumerate(marker[2:19,]):
        trigger = r[1]
        position = r[0]
        
        for shifts in xrange(shiftbegin,nshifts):
            position = position + shifts*windowsize
            ind = 0
            for item in BANDS :
                start = np.floor(position/Fraction) 
                end = np.floor(position/Fraction) + LenSig
                if end > dwsampleData[item].shape[0]:
                    continue;
                chopSignals = dwsampleData[item][start:end,:]               

                data_mat[0, index, :,ind:ind+BandROINUM[item]] = zscore(chopSignals,axis=0)
                ind += BandROINUM[item]
                
            label_mat[0, index] = trigger
            index += 1
               
            
    return data_mat, label_mat

def loadData_LikeDislike(File=None,
                         datadir= None  ,
                         markerdir = None,
                         LABELS = None,
                         roi_n = None ,
                         LenSig = None,
                         n_label = None,
                         BANDS = None,
                         ClassesNames=None,
                         tstring=None):

    if datadir is None or markerdir is None or LABELS is None or \
       roi_n is None or LenSig is None or n_label is None or BANDS is None:
        print 'Not enough input! ...'
        return 
    
    if File is None:
        marker_list = []
        data_list = []
        
        for item in os.listdir(datadir):
            data_list.append(item)
            marker_list.append(item)
    
            idxs = np.arange(len(data_list))
            idxs_train, idxs_test = train_test_split(idxs, test_size=0.1)
            idxs = np.arange(idxs_train.shape[0])
            idxs_train2, idxs_test2 = train_test_split(idxs, test_size=0.1)
            idxs_validation = idxs_train[idxs_test2]
            idxs_train = idxs_train[idxs_train2]  
            mat = {'data_list':data_list,'marker_list':marker_list,'idxs_train':idxs_train,
                   'idxs_test':idxs_test,'idxs_validation':idxs_validation}
        with open('TrainData'+ClassesNames+tstring+'.pickle','w') as f:
            cPickle.dump(mat,f)
    else:
        with open(File,'r') as f:
            mat = cPickle.load(f)
        data_list = list(mat['data_list'])
        marker_list = list(mat['marker_list'])
        idxs_train = np.asarray(mat['idxs_train'])
        idxs_test = np.asarray(mat['idxs_test'])
        idxs_validation = np.asarray(mat['idxs_validation'])  

        
    data_mat_list = []
    label_mat_list = []

    for data_file,marker_file in  zip(data_list,marker_list):
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat, label_mat = load_mat_likeDislike(data,
                                                   marker, 
                                                   roi_n=roi_n, 
                                                   LenSig=LenSig,
                                                   n_label=n_label,
                                                   BANDS = BANDS)   
        data_mat_list.append(data_mat)
        label_mat_list.append(label_mat) 
        
    data_mat_t = np.concatenate(data_mat_list, axis=0)
    label_mat_t = np.concatenate(label_mat_list, axis=0)
    
    wherenan = np.isnan(data_mat_t)
    whereinf = np.isinf(data_mat_t)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat_t[nanidx] = 0
    data_mat_t[infidx] = 0

    train_data = data_mat_t[idxs_train, ]
    test_data = data_mat_t[idxs_test, ]
    train_label = label_mat_t[idxs_train, ]
    test_label = label_mat_t[idxs_test, ]
    val_data = data_mat_t[idxs_validation, ]
    val_label = label_mat_t[idxs_validation, ]    
    train_data_flat = np.reshape(train_data, (-1, LenSig, roi_n))
    train_label_flat = np.reshape(train_label, (-1))
    val_data_flat = np.reshape(val_data, (-1, LenSig, roi_n))
    val_label_flat = np.reshape(val_label, (-1))    
    test_data_flat = np.reshape(test_data, (-1, LenSig, roi_n))
    test_label_flat = np.reshape(test_label, (-1))


    X_train = train_data_flat[:]
    y_train = train_label_flat[:]
    X_val = val_data_flat[:]
    y_val = val_label_flat[:]
    X_test = test_data_flat[:]
    y_test = test_label_flat[:]
    
    idx = np.where(np.in1d(y_train, LABELS)  )[0]
    y_train = y_train[idx]
    X_train = X_train[idx]
    y_tmp = np.zeros((y_train.shape[0],len(LABELS)))
    for ix,i in enumerate(y_train):
        y_tmp[ix,LABELS.index(i)] = 1
    y_train = np.copy(y_tmp)
    
    
    idx = np.where(np.in1d(y_test, LABELS) )[0]
    y_test = y_test[idx]
    X_test = X_test[idx]
    y_tmp = np.zeros((y_test.shape[0],len(LABELS)))
    for ix,i in enumerate(y_test):
        y_tmp[ix,LABELS.index(i)] = 1
    y_test = np.copy(y_tmp) 
    
    
    idx = np.where(np.in1d(y_val, LABELS) )[0]
    y_val = y_val[idx]
    X_val = X_val[idx]
    y_tmp = np.zeros((y_val.shape[0],len(LABELS)))
    for ix,i in enumerate(y_val):
        y_tmp[ix,LABELS.index(i)] = 1
    y_val = np.copy(y_tmp)    
    
    return X_train,y_train,X_val,y_val,X_test,y_test



def GET_FRAMES_LIKEDISLIKE(data,marker,LenSig,BANDS):
    Fraction= 4.0
    FrameRate = 15
    shiftSecods = np.ceil(4.5*FrameRate)
    Window = LenSig
    marker = np.ceil(marker[:,0]/Fraction)
    nChanel = len(BANDS)
    
    dataLength = data['data'].shape[1]
    dataChopLength = marker[19] - marker[1]
    numFrames = np.ceil(dataChopLength/shiftSecods)
    
    # down sampling from 60 Hz to 15 Hz
    data_mat = np.zeros(( numFrames,nChanel, LenSig,9))
    dwsampleData = {}
    for item in BANDS:
        dwsampleData[item] = np.zeros((np.ceil(data[item].shape[1]/Fraction),9))
        for q in xrange(9):
            dwsampleData[item][:,q]= decimate(data[item][q,:], int(Fraction))     

   
              
    # after down sample set the data length    
    dataLength = np.ceil(dataLength/Fraction)    
    FramesStart = np.arange(marker[1],marker[19],shiftSecods)
    index  = 0
    for start in FramesStart: 
        end = start + LenSig             
        ind = 0
        for item in BANDS :
            if end > dwsampleData[item].shape[0]:
                continue
            chopSignals = dwsampleData[item][start:end,:]
            data_mat[index,ind, :, :] = zscore(chopSignals,axis=0)
            ind += 1  
        index +=1
    
    
    data_mat = np.transpose(data_mat,axes=(0,2,1,3))
    data_mat = data_mat.reshape(data_mat.shape[0],LenSig,9*nChanel)
    
    wherenan = np.isnan(data_mat)
    whereinf = np.isinf(data_mat)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat[nanidx] = 0
    data_mat[infidx] = 0        
            
    return data_mat 

def GET_FRAMES(data,marker):
    Fraction= 4.0
    FrameRate = 15
    shiftSecods = np.ceil(4.5*FrameRate)
    Window = LenSig
    marker = np.ceil(marker[:,0]/Fraction)
    nChanel = len(BANDS)
    
    dataLength = data['data'].shape[1]
    dataChopLength = marker[29] - marker[1]
    numFrames = np.ceil(dataChopLength/shiftSecods)
    
    # down sampling from 60 Hz to 15 Hz
    data_mat = np.zeros(( numFrames,nChanel, LenSig,9))
    dwsampleData = {}
    for item in BANDS:
        dwsampleData[item] = np.zeros((np.ceil(data[item].shape[1]/Fraction),9))
        for q in xrange(9):
            dwsampleData[item][:,q]= decimate(data[item][q,:], int(Fraction))     

   
              
    # after down sample set the data length    
    dataLength = np.ceil(dataLength/Fraction)    
    FramesStart = np.arange(marker[1],marker[29],shiftSecods)
    index  = 0
    for start in FramesStart: 
        end = start + LenSig             
        ind = 0
        for item in BANDS :
            if end > dwsampleData[item].shape[0]:
                continue
            chopSignals = dwsampleData[item][start:end,:]
            data_mat[index,ind, :, :] = zscore(chopSignals,axis=0)
            ind += 1  
        index +=1
    
    
    data_mat = np.transpose(data_mat,axes=(0,2,1,3))
    data_mat = data_mat.reshape(data_mat.shape[0],LenSig,9*nChanel)
    
    wherenan = np.isnan(data_mat)
    whereinf = np.isinf(data_mat)
    nanidx = np.where(wherenan==True)
    infidx = np.where(whereinf==True)
    data_mat[nanidx] = 0
    data_mat[infidx] = 0        
            
    return data_mat 


def GET_LAST_LAYER_REP(fileid):
    datadir = '../Emovid600/bloodflow_bands/'
    markerdir = '../Emovid600/rateMarker/'
    
    modelFile = 'best_model'+ ClassesNames + fileid + '.hdf5'
    encoder = loadCNNEncoder(modelFile) 
   
    marker_list = []
    data_list = []

    for item in os.listdir(datadir):
        data_list.append(item)
        marker_list.append(item)

    data_mat_list = []
    label_mat_list = []   
    
    for data_file,marker_file in  zip(data_list,marker_list):
        print data_file, ' ...........  '
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat = GET_FRAMES(data, marker) 
        data_mat = encoder.predict(data_mat,batch_size=data_mat.shape[0])
        mat = {'data':data_mat.T}
        savemat('./FEATURES_'+fileid+'/'+data_file,mat)
    


def GET_LAST_LAYER_REP_LIKEDISLIKE(encoder,LenSig,BANDS):
    datadir = '../NuralogixLikeDislike/bloodflow_bands/'#'../Emovid600/bloodflow_bands/'
    markerdir = '../NuralogixLikeDislike/rateMarker/'#'../Emovid600/rateMarker/'
     
   
    marker_list = []
    data_list = []

    for item in os.listdir(datadir):
        data_list.append(item)
        marker_list.append(item)

    data_mat_list = []
    label_mat_list = []   
    
    for data_file,marker_file in  zip(data_list,marker_list):
        print data_file, ' ...........  '
        marker = loadmat(os.path.join(markerdir, marker_file))['subdata']
        data = loadmat(os.path.join(datadir, data_file))
        data_mat = GET_FRAMES_LIKEDISLIKE(data, marker,LenSig,BANDS) 
        data_mat = encoder.predict(data_mat,batch_size=data_mat.shape[0])
        mat = {'data':data_mat.T}
        savemat('./NuraLogixLikeDislike_FEATURES/'+data_file,mat)
    

        
import argparse
import torch
import wandb
import numpy as np
import tifffile as tiff
import glob
from PIL import Image
from osgeo import gdal
import keras
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("train device:", device)

def verify(args):  
    wandb.init(project='test2', name='test2')
    '''
    # print out channels for verification purposes (see in stdout log file)
    src_ds = gdal.Open("/home/c_dl_bm/banmarton/test2/3band/Train/image/Chongqing_1.tif")
    if src_ds is not None:
        print ("3 band count: " + str(src_ds.RasterCount))

    src_ds = gdal.Open("/home/c_dl_bm/banmarton/test2/4band/Train/image/Chongqing_1.tif")
    if src_ds is not None:
        print ("4 band count: " + str(src_ds.RasterCount))

    # connect to wandb and print out images and labels to verify visually the data:
    img3band = tiff.imread('/home/c_dl_bm/banmarton/test2/3band/Train/image/Chongqing_1.tif')
    img3bandLabel = tiff.imread('/home/c_dl_bm/banmarton/test2/3band/Train/label/Chongqing_1.tif') 

    
    wandb.log({"image 3band":wandb.Image(img3band)})
    wandb.log({"label 3band":wandb.Image(img3bandLabel)})

    img4band = tiff.imread('/home/c_dl_bm/banmarton/test2/4band/Train/image/Chongqing_1.tif')    
    img4bandLabel = tiff.imread('/home/c_dl_bm/banmarton/test2/4band/Train/label/Chongqing_1.tif')    
    wandb.log({"image 4band":wandb.Image(img4band)})
    wandb.log({"label 4band":wandb.Image(img4bandLabel)})
    '''  
    
    #load images
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Train/image/*.tif')
    Xtrain = np.array([np.array(Image.open(fname)) for fname in filelist])
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Test/image/*.tif')
    Xtest = np.array([np.array(Image.open(fname)) for fname in filelist])
    print("Shape Xtrain:", np.shape(Xtrain))
    print("Shape Xtest:", np.shape(Xtest))

    #load labels
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Train/label/*.tif')
    ytrain = np.array([np.expand_dims(Image.open(fname), axis=2) for fname in filelist])  
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Test/label/*.tif')
    ytest = np.array([np.expand_dims(Image.open(fname), axis=2) for fname in filelist])
    print("Shape ytrain:", np.shape(ytrain))
    print("Shape ytrain:", np.shape(ytest))

    wandb.log({"Images 4band":[wandb.Image(x_t) for x_t in Xtrain[:7]]})
    wandb.log({"Labels 4band":[wandb.Image(x_t) for x_t in ytrain[:7]]})

    # for binary cross enthropy we need onehot like 2 labels:
    #ytrain= np.squeeze(ytrain, axis=3) # (None, 512, 512, 1, 2) -> (None, 512, 512, 2)
    #ytest= np.squeeze(ytest, axis=3)
    print("Shape ytrain before squeeze:", np.shape(ytrain))
    print("Shape ytrain before squeeze:", np.shape(ytest))

    ytrain = tf.squeeze(tf.one_hot(ytrain, depth=2), axis=3) # (None, 512, 512, 1, 2) -> (None, 512, 512, 2)
    ytest = tf.squeeze(tf.one_hot(ytest, depth=2), axis=3) 
    #ytrain = torch.nn.functional.one_hot(ytrain, 2).transpose(1, 4).squeeze(-1)
    #ytest = torch.nn.functional.one_hot(ytrain, 2).transpose(1, 4).squeeze(-1)

    #ytrain = np.ones((2, 512, 512))
    #ytrain = torch.from_numpy(ytrain)

    #ytest = np.ones((2, 512, 512))
    #ytest = torch.from_numpy(ytest)

    print("OHE Shape ytrain:", tf.shape(ytrain))
    print("OHE Shape ytrain:", tf.shape(ytest))
   
    #normalize
    Xtrain = Xtrain/255
    Xtest = Xtest/255

    # metrics:
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    #load h5 file and fit dataset    
    #model = load_model('/home/c_dl_bm/banmarton/test2/models/UNet.h5') 
    model = load_model('/home/c_dl_bm/banmarton/test2/models/UNet.h5', compile=False)         
    wandb.log({"model summary": model.summary()})

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()])
    history = model.fit(Xtrain, ytrain, validation_split=0.25, epochs=10, verbose=1)

    wandb.log({"finished": "Program finished successfully"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    verify(args)
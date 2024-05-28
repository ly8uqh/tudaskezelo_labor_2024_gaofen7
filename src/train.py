import argparse
import torch
import wandb
import numpy as np
import glob
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("train device:", device)

def verify(args):  
    wandb.init(project='test2', name='test2')
    
    #load images
    #filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Train/image/*.tif')
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Val/image/*.tif')
    Xtrain = np.array([np.array(Image.open(fname)) for fname in filelist])
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Test/image/*.tif')
    Xtest = np.array([np.array(Image.open(fname)) for fname in filelist])
    print("Shape Xtrain:", np.shape(Xtrain))
    print("Shape Xtest:", np.shape(Xtest))

    #load labels
    #filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Train/label/*.tif')
    filelist = glob.glob('/home/c_dl_bm/banmarton/test2/4band/Val/label/*.tif')
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

    #paths = os.path('/home/c_dl_bm/banmarton/test2/models*.h5')
    paths = glob.glob("/home/c_dl_bm/banmarton/test2/models/*.h5")
    print("paths: ", paths)

    for counter, file in enumerate(paths):
        print('Starting evaluating model: ', file)        
        new_model = load_model(file, compile=False)         
        #wandb.log({"model summary": new_model.summary()})
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()]) 
        loss = new_model.evaluate(Xtrain, ytrain, verbose=2)
        print('Restored model, loss: ', loss)
    
    
    wandb.log({"finished": "Program finished successfully"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    verify(args)
    

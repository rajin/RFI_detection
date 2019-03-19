### loading the keras libraries
from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import History 

from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np 
import h5py
from sklearn.utils import shuffle


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def model_LSTM():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         input_shape=(None, 6, 4096, 1), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last')) 

    print(model.summary())
    return model


def fn_run_model(model, X, y, X_val, y_val, batch_size=50, nb_epoch=40,verbose=2,is_graph=False):
    history = History()
    history = model.fit(X, y, batch_size=batch_size, 
                        epochs=nb_epoch,verbose=verbose, validation_data=(X_val, y_val))
    if is_graph:
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(history.history["val_loss"])
        ax1.plot(history.history["loss"])

model = model_LSTM()
model.compile(loss='mean_squared_error', optimizer='adam')


def open_h5file(filename):
    #function to read HDF5 file format
    return h5py.File(filename,'r')

dirty_seq = open_h5file('dirty_seq.h5')
rfi_seq = open_h5file('rfi_seq.h5')

dirty_seq = dirty_seq['dirty_seq'].value
rfi_seq = rfi_seq['rfi_seq'].value

r_dirty_seq = np.reshape(dirty_seq,(784,8,6,4096,1))
r_rfi_seq = np.reshape(rfi_seq,(784,8,6,4096,1))


X,Y = shuffle(r_dirty_seq,r_rfi_seq,random_state=0)

X_train = X[:700,:7,:,:]
X_val = X[700:,:7,:,:]
Y_train = Y[:700,7,:,:]
Y_val = Y[700:,7,:,:]

fn_run_model(model, X_train, Y_train, X_val, Y_val, batch_size=1, nb_epoch=15,verbose=1,is_graph=True)


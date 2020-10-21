from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, MaxPooling2D, Activation
import pandas as pd
import numpy as np
import os, sys
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras import preprocessing
def normalize(matrix):
    matrix[:,:-1] =  (matrix[:,:-1]-np.min(matrix[:,:-1]))/(np.max(matrix[:,:-1]) - np.min(matrix[:,:-1]))
    matrix[:,-1:] = (matrix[:,-1:]-np.min(matrix[:,-1:]))/(np.max(matrix[:,-1:]) - np.min(matrix[:,-1:]))
    return matrix


def baseline_model():
    model = Sequential()
    model.add(LSTM(units=400))
    model.add(Dropout(0.3))
    #model.add(Dense(4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



######################################################################
# read in datasets
def createModel(dataDir, savefile, epochs, batchsize):
    dir = dataDir
    sets = os.listdir(dir)

    if len(sets) > 0:
        # Initialize Data
        data = pd.read_csv(dir + '\\' + sets[0])
        train_size = int(.8*data.shape[0]) # 80% of data for training
        x = data.iloc[:,1:6].to_numpy() # y = data.loc[1:,'Close']
        x = normalize(x)

        # Parse data
        x_new_train = []
        y_new_train = []
        for i in range(60, x.shape[0]-30):
            x_new_train.append(x[i-60:i, :])
            y_new_train.append(np.average(x[i:i+30,3]) > np.average(x[i,3]))
        x_train, y_train = np.array(x_new_train), np.array(y_new_train)
        y_train = np.where(y_train > 0, 1, 0)
        x_test, y_test = x_train[train_size:, :], y_train[train_size:]
        x_train, y_train = x_train[:train_size, :], y_train[:train_size]

        sets.pop(0)
    else:
        return

    for i in range(100):
       try:
            # Initialize Data
            data = pd.read_csv(dir +'\\'+ sets[i])
            train_size = int(.8*data.shape[0])
            x = data.iloc[:,1:6].to_numpy()
            x = normalize(x)

            # Parse Data
            x_new_train = []
            y_new_train = []
            for i in range(60, x.shape[0]-30):
                x_new_train.append(x[i - 60:i, :])
                y_new_train.append(np.average(x[i:i+30,3]) > np.average(x[i:i,3]))

            x_tr, y_tr = np.array(x_new_train), np.array(y_new_train)
            y_tr = np.where(y_tr > 0, 1, 0)
            x_te, y_te = x_tr[train_size:, :], y_tr[train_size:]
            x_tr, y_tr = x_tr[:train_size, :], y_tr[:train_size]

            # accumulate data
            x_train = np.append(x_train, x_tr, axis=0)
            x_test = np.append(x_test, x_te, axis=0)

            y_train = np.append(y_train, y_tr, axis=0)
            y_test = np.append(y_test, y_te, axis=0)
       except:
           continue

    # Test and save model
    model = baseline_model()
    model.fit(x_train, y_train, epochs=epochs,batch_size=batchsize,verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("LSTM Error: %.2f%%" % (100-scores[1]*100))
    model.save(savefile)
    return


dir = 'C:\\Users\\Chand\\Desktop\\IndependentProjects\\Learning\\Machine_Learning\\archive\\Stocks'
savefile = 'modelbinary.h5'
epochs = 3
batch = 200
createModel(dir, savefile,epochs,batch)
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

def predict(model, dataPath):
    model = load_model(model)
    model.compile(loss='mse', optimizer='adam', metrics=['mape'])
    data = pd.read_csv(dataPath)
    x = data.iloc[:,1:6].to_numpy()
    print(x[-1:,3])
    # Normalize
    scale_price = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_vol = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scale_ref = preprocessing.MinMaxScaler(feature_range=(0, 1))
    close_price = np.reshape(x[:, 1], (-1, 1))
    scale_ref.fit_transform(close_price)
    x[:, :-1] = scale_price.fit_transform(x[:, :-1])
    x[:, -1:] = scale_vol.fit_transform(x[:, -1:])

    # Get Data
    x_new_train = []
    for i in range(30, x.shape[0] - 30):
        x_new_train.append(x[i - 30:i, :])
    x_train = np.array(x_new_train)

    # Predict and output
    score = model.predict(x_train[-1:,])
    #score = score[0]
    print(scale_ref.inverse_transform(np.array([x[-1:,3]])))
    price = score + 10*x[-1:, 3]
    print(price)

    price = scale_ref.inverse_transform(price)
    print(score + 9*x[-1:, 3])
    purchase = score + 9*x[-1:, 3] > .2


    percent_change = (price - x[-1:,3])/ x[-1:,3]
    print('Estimate 20-30 day avg: %.2f' % price)
    print('Estimate 20-30 percent change: %.2f%%' % percent_change)
    if purchase:
        print("This is a good purchase")
    else:
        print("Purchase with caution")




def main():
    if len(sys.argv) > 2:
        raise ValueError('Too many arguments.')
    if len(sys.argv) < 2:
        raise ValueError('Too few arguments.')
    modelPath = sys.argv[1]
    dataPath = sys.argv[2]
    predict(modelPath, dataPath)

datapath = 'C:\\Users\\Chand\\Desktop\\IndependentProjects\\Learning\\Machine_Learning\\verizon_test'
model = 'C:\\Users\\Chand\\Desktop\\IndependentProjects\\Learning\\Machine_Learning\\200lstm100.h5'
predict(model, datapath)

#if __name__ == '__main__':
 #   main()


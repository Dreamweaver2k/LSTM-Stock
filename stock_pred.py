from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


def transform(matrix, valid_number, normalization):
    matrix = matrix*normalization/np.max(scores[:valid_number])
    return matrix


def baseline_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


######################################################################
# read in datasets
dir = 'C:\\Users\\Chand\\Desktop\\IndependentProjects\\Learning\\Machine_Learning\\archive\\Stocks'
sets = os.listdir(dir)

# Initialize
data = pd.read_csv(dir + '\\' + sets[0])
train_size = int(.8*data.shape[0]) # 80% of data for training

x = data.iloc[:,1:6].to_numpy() # y = data.loc[1:,'Close']

# Training data
#x_train = x.iloc[:train_size, :].to_numpy()
price_norm = np.max(x[:,:-1])
vol_norm = np.max(x[:,-1:])

x[:,:-1] = x[:,:-1]/price_norm
x[:,-1:] = x[:,-1:]/vol_norm

x_new_train = []
y_new_train = []
for i in range(60, x.shape[0]-1):
    x_new_train.append(x[i-60:i, :])
    y_new_train.append(x[i,3])
x_train, y_train = np.array(x_new_train), np.array(y_new_train)

x_test, y_test = x_train[train_size:, :], y_train[train_size:]
x_train, y_train = x_train[:train_size, :], y_train[:train_size]

sets.pop(0)

for i in range(2):
    #print(sets[i])
    try:
        data = pd.read_csv(dir +'\\'+ sets[i])
        train_size = int(.8*data.shape[0])

        # initialize data
        x = data.iloc[:,1:6].to_numpy()
        # normalize
        price_norm = np.max(x[:, :-1])
        vol_norm = np.max(x[:, -1:])

        x[:, :-1] = x[:, :-1] / price_norm
        x[:, -1:] = x[:, -1:] / vol_norm

        x_new_train = []
        y_new_train = []
        for i in range(60, x.shape[0] - 1):
            x_new_train.append(x[i - 60:i, :])
            y_new_train.append(x[i, 3])

        x_tr, y_tr = np.array(x_new_train), np.array(y_new_train)
        x_te, y_te = x_tr[train_size:, :], y_tr[train_size:]
        x_tr, y_tr = x_tr[:train_size, :], y_tr[:train_size]

        # accumulate data
        x_train = np.append(x_train, x_tr, axis=0)
        x_test = np.append(x_test, x_te, axis=0)

        y_train = np.append(y_train, y_tr, axis=0)
        y_test = np.append(y_test, y_te, axis=0)
    except:
        continue

# reshape
#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))

model = baseline_model()
model.fit(x_train, y_train, epochs=2,batch_size=1,verbose=2)

"""train = pd.read_csv(dir +'\\'+ sets[101])
x = train.iloc[:-1,1:5]
y = train.loc[1:,'Close'] - train.loc[1:,'Open']
x_test = x.iloc[:,:]
y_test = y.iloc[:]
x_test = x_test.to_numpy()
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))"""

print('made it')
scores = model.evaluate(x_test, y_test, verbose=2)
print(scores)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


actual_test = pd.read_csv('C:\\Users\\Chand\\Desktop\\IndependentProjects\\Learning\\Machine_Learning\\datasets\\stocks\\gs_n.us.txt')
x = actual_test.iloc[:-1,1:6].to_numpy()
prime = 100
price_norm = np.max(x[:prime,:-1])
vol_norm = np.max(x[:prime,-1:])
print(price_norm)

x[:,:-1] = x[:,:-1]/price_norm
x[:,-1:] = x[:,-1:]/vol_norm


x_new_train = []
y_new_train = []
for i in range(60, x.shape[0]):
    x_new_train.append(x[i-60:i, :])
    y_new_train.append(x[i,3])

x, y = np.array(x_new_train), np.array(y_new_train)

scores = model.predict(x)

scores = transform(scores, prime, price_norm)
plt.plot(scores)
plt.plot(y * price_norm)
plt.plot(np.average(y))
plt.show()

# print()
model.save('model5.h5')
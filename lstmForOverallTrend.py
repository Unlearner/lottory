

####
####use lstm to predict the overall trend of lotto numbers
from sklearn.preprocessing import MinMaxScaler
##normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
look_back = 16
path = "/Users/marsly/data.csv"
import loadDataFromCsv
dataset = loadDataFromCsv.load_data_from_csv(path)

dataset_red_x, dataset_red_y = loadDataFromCsv.data_sum_by_blue(dataset,look_back)
print(dataset_red_x.shape)
print(dataset_red_y.shape)

x = dataset_red_x[:,1:(look_back+1)]/24
y = dataset_red_y/24

print(x.shape)

train_size = int(len(x)*0.7)
test_size  = len(x) - train_size

train_x, train_y, test_x,test_y = x[0:train_size,:],y[0:train_size],x[train_size:len(x),:],y[train_size:len(x)]
import numpy
train_x = numpy.reshape(train_x,(train_x.shape[0],1,look_back))
test_x  = numpy.reshape(test_x,(test_x.shape[0],1,look_back))

print(train_x.shape)
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=2000, batch_size=5, verbose=2)


trainPredict = model.predict(train_x)
testPredict = model.predict(test_x)

import matplotlib.pyplot as plt

plt.plot(test_y*24)
plt.plot(testPredict*24)
plt.show()

def main():
    return
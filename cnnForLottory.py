###主要是利用cnn的方法来预测lottory
###得到自相关矩阵
def auto_correlation_matrix(v):
    l   = v.shape[0]
    mat = np.zeros([l, l])
    for i in range(0, l):
        for j in range(0, l):
            mat[i,j] = v[i]*v[j]

    return mat

#####将矩阵变成0, 1组成的向量
def to_one_zero_vector(v, len):
    one_zero_v = np.zeros(len)
    for i in range(0, v.shape[0]):
        one_zero_v[v[i]-1] = 1
    return one_zero_v


path = "/Users/marsly/data.csv"
import loadDataFromCsv
dataset = loadDataFromCsv.load_data_from_csv(path)

import numpy as np

red_len = 35
blue_len= 12
m = np.zeros([35,35])

hist_red = np.zeros(red_len)
hist_blue= np.zeros(blue_len)
col_len = dataset.shape[0]
x_red, y_red, x_blue, y_blue, pred_x_red, pred_x_blue = [],[],[],[],[],[]

for i in range(0,col_len):
    #红色球部分
    red_sub_data = dataset[col_len - i - 1,1:6]
    for j in range(0, 5):
        hist_red[red_sub_data[j]-1] = hist_red[red_sub_data[j]-1] + 1
    #蓝色球部分
    blue_sub_data= dataset[col_len - i - 1,6:8]
    for j in range(0, 2):
        hist_blue[blue_sub_data[j]-1]= hist_blue[blue_sub_data[j]-1] + 1
    ###生成训练用的数据
    if i < col_len-1:
        y_red.append(to_one_zero_vector(red_sub_data,red_len))
        y_blue.append(to_one_zero_vector(blue_sub_data, blue_len))
        x_red.append(auto_correlation_matrix(hist_red / (max(hist_red))))
        x_blue.append(auto_correlation_matrix(hist_blue / (max(hist_blue))))
    else:
        pred_x_red.append(auto_correlation_matrix(hist_red / (max(hist_red))))
        pred_x_blue.append(auto_correlation_matrix(hist_blue / (max(hist_blue))))
###生成可训练的形式
x_red = np.array(x_red).reshape(-1, red_len, red_len, 1)
y_red = np.array(y_red)

x_blue= np.array(x_blue).reshape(-1, blue_len, blue_len, 1)
y_blue= np.array(y_blue)

###将数据拆分成训练集和测试集
from sklearn.model_selection import train_test_split
x_red_train, x_red_val, y_red_train, y_red_val =  train_test_split(x_red, y_red, test_size = 0.2, random_state=11)
x_blue_train, x_blue_val, y_blue_train, y_blue_val =  train_test_split(x_blue, y_blue, test_size = 0.2, random_state=11)

###基于keras构建cnn模型
from  keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(filters= 4, kernel_size = (3,3), padding = 'same',
                 activation = 'relu', input_shape = (35, 35, 1)))
model.add(Conv2D(filters= 4, kernel_size = (3,3), padding = 'same',
                 activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(70, activation = 'relu'))
model.add(Dense(35, activation = 'relu'))

###参数迭代
optimizer = RMSprop(lr=1e-07, rho=0.91, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy",
              metrics=["accuracy"])


model.fit(x_red_train, y_red_train, epochs=200, batch_size=4, verbose=2)
red_pred_res = model.predict(x_red_val)

def prediction_stat(pred, actual, top_num, num ):
    res = []
    for i in range(0, pred.shape[0]):
        tmp   = pred[i,]
        index = np.argsort(tmp)
        tmp[index[-top_num:]]     = 1
        tmp[index[0:num-top_num]] = 0
        res.append(top_num - sum(abs(tmp - actual[i,]))/2)
    return res

###进行测试结果统计看预测对几个
pp = prediction_stat(red_pred_res, y_red_val, 5, 35)

def get_pred_stat(v, num):
    res = {}
    for i in range(0, num + 1):
        res[i] = v.count(i)
    return res


print("红球的预测结果: " + str(get_pred_stat(pp,5)))


###蓝色球的预测
model_blue = Sequential()
model_blue.add(Conv2D(filters= 4, kernel_size = (3,3), padding = 'same',
                 activation = 'relu', input_shape = (blue_len, blue_len, 1)))
model_blue.add(Conv2D(filters= 4, kernel_size = (3,3), padding = 'same',
                 activation = 'relu'))
model_blue.add(MaxPool2D(pool_size = (2,2)))

model_blue.add(Flatten())
model_blue.add(Dense(24, activation = 'relu'))
model_blue.add(Dense(12, activation = 'relu'))

###参数迭代
optimizer = RMSprop(lr=1e-07, rho=0.91, epsilon=1e-08, decay=0.0)
model_blue.compile(optimizer = optimizer, loss = "categorical_crossentropy",
              metrics=["accuracy"])


model_blue.fit(x_blue_train, y_blue_train, epochs=200, batch_size=4, verbose=2)
blue_pred_res = model_blue.predict(x_blue_val)

pp_blue = prediction_stat(blue_pred_res, y_blue_val, 2, 12)
print("红球的预测结果: " + str(get_pred_stat(pp_blue,2)))

print("所有球的预测结果: " + str(get_pred_stat(pp_blue + pp,7)))

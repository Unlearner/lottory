####load data from csv
import numpy
from pandas import read_csv
csv_path = "/Users/marsly/data.csv"
def load_data_from_csv(path):
    dataframe = read_csv(path,engine='python',header=None)
    # print(dataframe.values[:,0:8])
    return (dataframe.values[:,0:8])

lotto_data = load_data_from_csv(csv_path)
def data_sum_by_red(df, look_back=1):
    df_red = df[:,1]+df[:,2]+df[:,3]+df[:,4]+df[:,5]
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        tmp = [df[i:(i+look_back),0][0]]
        df_sub = df_red[i+1:(i + look_back+1)]
        for j in range(look_back):
            tmp.append(df_sub[j])
        dataX.append(tmp)
        dataY.append(df_red[i])
    return numpy.array(dataX), numpy.array(dataY)

def data_sum_by_blue(df, look_back=1):
    df_blue = df[:,6]+df[:,7]
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        tmp = [df[i:(i+look_back),0][0]]
        df_sub = df_blue[i+1:(i + look_back+1)]
        for j in range(look_back):
            tmp.append(df_sub[j])
        dataX.append(tmp)
        dataY.append(df_blue[i])
    return numpy.array(dataX), numpy.array(dataY)
#
# X,Y = data_sum_by_blue(lotto_data)
# print(X)


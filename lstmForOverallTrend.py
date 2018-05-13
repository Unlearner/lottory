####
####use lstm to predict the overall trend of lotto numbers
from sklearn.preprocessing import MinMaxScaler
##normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))

path = "/Users/marsly/data.csv"
import loadDataFromCsv
dataset = loadDataFromCsv.load_data_from_csv(path)
dataset_red_x, dataset_red_y = loadDataFromCsv.data_sum_by_blue(dataset,3)
print(dataset_red_x)
print(dataset_red_y)

x = dataset_red_x[:,1]
y = dataset_red_y

import tensorflow
def main():
    return
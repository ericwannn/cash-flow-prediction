import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold, GridSearchCV


data = np.genfromtxt('features.csv', delimiter=',')
target = np.genfromtxt('target.csv')
target = target[7:]

# mean_of_data = np.mean(data[:,2])
# std_of_data = np.std(data[:,2])

# mean_of_target = np.mean(target)
# std_of_target = np.std(target)

# data[:,2] = (data[:, 2] - mean_of_data) / std_of_data
# target = (target - mean_of_target) / std_of_target

X_train, X_test,  y_train, y_test  = train_test_split(data, target, test_size=0.1, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

def show_reg(iy1, iy2):
    plt.figure(figsize=(20, 10))
    ix = range(len(iy1))
    plt.plot(ix, iy1, label='true')
    plt.plot(ix, iy2, label='predict')
    plt.legend()
    plt.show()

data_train = xgb.DMatrix(X_train, label=y_train)
data_valid = xgb.DMatrix(X_valid, label=y_valid)
data_test  = xgb.DMatrix(X_test,  label=y_test)

param = {
    'max_depth': 10, 
    'eta': 0.1, 
    'lambda': 0.6,
    'objective': 'reg:linear'
    }  

watch_list = [(data_valid, 'eval'), (data_train, 'train')]

bst = xgb.train(
    param, data_train, 
    num_boost_round=2000, 
    evals=watch_list
)  
y_preds = bst.predict(data_test)

show_reg(y_test, y_preds)
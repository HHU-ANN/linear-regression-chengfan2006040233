# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    a = 7
    w = np.matmul(np.linalg.inv(np.matmul(X.T,X)+a*np.eye(X.shape[1])),np.matmul(X.T,y))
    return w@data

def lasso(data):
    X, y = read_data()
    w = np.zeros((X.shape[1], 1))
    limit = 2e-5
    a = 0.01
    step = 0.00000001
    epochs = 1000
    for i in range(epochs):
        loss = np.matmul((X - y).transpose(), X - y) + a * np.sum(abs(w))
        if loss < limit:
            break
        dw = np.matmul(X - y, X) + np.sign(w)
        w = w - step * dw
    return w@data
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

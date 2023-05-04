# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
import sympy as sy

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
   X,y = read_data()
   a = 4
   w = np.matmul(np.linalg.inv(np.matmul(X.T,X)+a*np.eye(X.shape[1])),np.matmul(X.T,y))
   return w@data


def lasso(data):
    x, y = read_data()
    miu = np.mean(x)
    sigma = np.std(x)
    for i in range(x.shape[0]):
        x[i] = (x[i] - miu) / sigma

    y = np.array([y]).transpose()
  
    lamda = 1
    a = 0.01
    epochs = 1000
    num = x.shape[0]
    xlen = x.shape[1]
    w, b = np.zeros((xlen, 1)), 0
    for _ in range(epochs):
        y_hat = np.dot(x, w) + b  

        dw = (np.dot(x.transpose(), (y_hat - y)) / num) + lamda * diff(w)
        db = np.sum(y_hat - y) / num
        w -= a * dw
        b -= a * db

    data = (data - miu) / sigma
    res = np.dot(w.transpose(), data) + b
    return res
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

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


def piandao(w):
    partial_l1 = [elem for elem in w]
    for i in range(w.shape[0]):
        if partial_l1[i] > 0:
            partial_l1[i] = 1
        elif partial_l1[i] < 0:
            partial_l1[i] = -1
        else:
            partial_l1[i] = 0
    partial_l1 = np.array([partial_l1]).transpose()
    return partial_l1


def lasso(data):
    x, y = read_data()
    miu = np.mean(x)
    sigma = np.std(x)
    for i in range(x.shape[0]):
        x[i] = (x[i] - miu) / sigma

    y = np.array([y]).transpose()

    lamda = 1
    step = 0.01
    epochs = 1000
    num = x.shape[0]
    xlen = x.shape[1]
    w, b = np.zeros((xlen, 1)), 0
    for _ in range(epochs):
        y_hat = np.dot(x, w) + b  # 404*1

        dw = (np.dot(x.transpose(), (y_hat - y)) / num) + lamda * piandao(w)
        db = np.sum(y_hat - y) / num

        w -= step * dw
        b -= step * db

    data = (data - miu) / sigma
    res = np.dot(w.transpose(), data) + b
    return res
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

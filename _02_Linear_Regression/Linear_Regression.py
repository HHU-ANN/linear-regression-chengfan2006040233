# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    a = -0.1
    w = np.dot(np.linalg.inv((np.dot(x.transpose(), x) + a* np.eye(6))), np.dot(x.transpose(), y))
    return w @ data


def lasso(data):
    x, y = read_data()
    w = np.array([0, 0, 0, 0, 0, 0])
    limit = 2e-5
    a = 0.0001
    step = 1e-12
    for i in range(int(2e6)):
        X = np.dot(x, w)
        loss = np.dot((X - y).transpose(), X - y) + a * np.sum(abs(w))
        if loss < limit:
            break
        dw = np.dot(X - y, x) + np.sign(w)
        w = w - step * dw
    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

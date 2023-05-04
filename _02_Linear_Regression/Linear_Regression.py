# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    i = np.eye(6)
    a = -0.1
    w = np.matmul(np.linalg.inv(np.matmul(x.transpose(), x) + np.matmul(a, i)), np.matmul(x.transpose(), y))
    print(np.matmula, i)
    return data @ w


def lasso(data):
    x, y = read_data()
    w = np.array([0, 0, 0, 0, 0, 0])
    limit = 2e-5
    a = 0.01
    step = 1e-12
    for i in range(int(2e6)):
        z = np.matmul(x, w)
        loss = np.matmul((z - y).transpose(), z - y) + a * np.sum(abs(w))
        if loss < limit:
            break
        dw = np.matmul(z-y, x) + np.sign(w)
        w = w - step * dw
    return data @ w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

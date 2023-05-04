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
    x, y = read_data()
    weight = np.array([0, 0, 0, 0, 0, 0])
    label = 2e-5
    alpha = 0.01
    r = 1e-12
    for i in range(int(2e6)):
        loss = np.dot((x - y).T, x - y) + alpha * np.sum(abs(weight))
        if loss < label:
            break
        dw = np.dot(x - y, x) + np.sign(weight)
        weight = weight - r * dw
    return data @ weight
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

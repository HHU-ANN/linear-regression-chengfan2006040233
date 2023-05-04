# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    a = 5
    w =  np.matmul(np.linalg.inv(np.matmul(X.T,X)+a*np.eye(X.shape[1])),np.matmul(X.T,y))
    return w@data
    
def lasso(data):
    X,y = read_data()
    n = X.shape[1]
    W=np.mat(np.ones((n,1)))
    Lambda = 1
    epochs = 1000
    a = 0.000001
    for i in range(epochs):
            gradient = X.T*(X*W-y)/m + Lambda * np.sign(W)
            W=W-a * gradient
        return W@data
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

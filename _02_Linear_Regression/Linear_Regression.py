# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    w =  np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
    return w@data
    
def lasso(data):
    x,y = read_data()
    #计算总数据量
    epochs = 1000
    Lambda = 0.0001
        m=x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        n = X.shape[1]
        W=np.mat(np.ones((n,1)))
        xMat = np.mat(X)
        yMat =np.mat(y.reshape(-1,1))
        for i in range(epochs):
            gradient = xMat.T*(xMat*W-yMat)/m + Lambda * np.sign(W)
            W=W-a * gradient
        return W
    def predict(self,x,w):  
        return np.dot(x,w)
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

import numpy as np
from .metrics import r2_score


class LinearRegression:
    def __init__(self):
        '''初始化 Linear Regression模型'''
        self.coef_ = None #系数
        self.interception_ = None #截距
        self._theta = None #theta整体矩阵


    def fit_normal(self, X_train, y_train):
        '''根据训练集X, y 训练Linear Regresssion模型'''
        assert X_train.shape[0] == y_train.shape[0], 'col must be equaled'

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) #构造一列全为1的矩阵
        self._theta  = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self


    def predict(self, X_predict):
        '''返回表示X_predict的结果向量'''
        assert self._theta is not None, 'must fit first'

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict]) #构造特征矩阵
        return X_b.dot(self._theta) #返回预测值



    def score(self, X_test, y_test):
        '''返回当前模型的准确度'''
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)



    def fit_gd(self, X_train, y_train , eta=0.01, n_iters=1e4):
        '''根据训练集，使用梯度下降法训练Linear Regression模型'''

        def J(theta, X_b, y):  # x_b已经增加了全为1的矩阵
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')  # 若超出界限则返回一个浮点数的最大值

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2.0 / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-10):
            theta = initial_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                i_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) #构造一列全为1的矩阵
        initial_theta= np.zeros(X_b.shape[1])
        self._theta  = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self




    def __repr__(self):
        return "Linear Regression()"

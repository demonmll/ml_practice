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





    def __repr__(self):
        return "Linear Regression()"

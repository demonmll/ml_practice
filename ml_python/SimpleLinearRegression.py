import numpy as np
from .metrics import r2_score

class SimpleLinearRegression1:

    def __init__(self):
        '''初始化simple linear regression 模型'''
        self.a_ =None ##使用下划线是因为用户没有传入参数，是系统自定义的参数
        self.b_ =None

    def fit(self, x_train, y_train): ##使用的是x，是因为简单线性规划是因为只有一个特征，x只是一个向量，而不是一个数组，因此不用X
        '''根据训练集x_train , y_train训练 simple linear regression模型'''
        assert x_train.ndim == 1, 'ndim == 1'
        assert len(x_train) == len(y_train)

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  ##分子 numerator
        d = 0.0  ##分母 denominator
        for x_i, y_i in zip(x_train, y_train):  ##zip方法一次打包一组数据
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self



    def predict(self, x_predict):
        '''预测给定数据集x_predict，返回表示x_predict的结果向量'''
        assert x_predict.ndim == 1, 'ndim == 1'

        return np.array([self._predict(x) for x in x_predict])



    def _predict(self, x_single):
        '''预测单个数据'''
        return self.a_* x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"



class SimpleLinearRegression2:

    def __init__(self):
        '''初始化simple linear regression 模型'''
        self.a_ =None ##使用下划线是因为用户没有传入参数，是系统自定义的参数
        self.b_ =None

    def fit(self, x_train, y_train): ##使用的是x，是因为简单线性规划是因为只有一个特征，x只是一个向量，而不是一个数组，因此不用X
        '''根据训练集x_train , y_train训练 simple linear regression模型'''
        assert x_train.ndim == 1, 'ndim == 1'
        assert len(x_train) == len(y_train)

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)  ##分子 numerator
        d = (x_train - x_mean).dot(x_train - x_mean)  ##分母 denominator

        # for x_i, y_i in zip(x_train, y_train):
        #     num += (x_i - x_mean) * (y_i - y_mean)
        #     d += (x_i - x_mean) ** 2




        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self



    def predict(self, x_predict):
        '''预测给定数据集x_predict，返回表示x_predict的结果向量'''
        assert x_predict.ndim == 1, 'ndim == 1'

        return np.array([self._predict(x) for x in x_predict])



    def _predict(self, x_single):
        '''预测单个数据'''
        return self.a_* x_single + self.b_


    def score(self, x_test, y_test):
        '''根据测试数据，返回准确度，调用的时r^2方法'''
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)




    def __repr__(self):
        return "SimpleLinearRegression2()"
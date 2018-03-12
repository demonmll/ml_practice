import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scaler_ = None

    def fit(self, X):
        '''根据训练数据集X获得数据的均值和方差'''
        assert X.ndim == 2, 'the dimension of X must be 2'

        self.mean_ =  np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scaler_ = np.std([np.mean(X[:,i]) for i in range(X.shape[1])])

        return self

    def tranform(self, X):
        '''将X根据方差和均值进行归一化'''
        assert X.ndim == 2, 'the dimension of X must be 2'
        assert self.mean_ is not None, 'must fit before transform'
        assert X.shape[1] == len(self.mean_), 'the feature number must be equal'


        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:,col] - self.mean_[col]) / self.scaler_[col]
        return resX

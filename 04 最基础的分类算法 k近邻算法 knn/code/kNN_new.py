import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:

    def __init__(self, k): ##初始化knn分类器
        assert k >= 1, "k must be valid"

        self.k = k
        self._X_train = None
        self._y_train = None



    def fit(self, X_train, y_train): ##根据训练集x_train和y_train训练knn分类器
        assert X_train.shape[0] == y_train.shape[0], "the sizes must be equaled"

        self._X_train = X_train
        self._y_train = y_train
        return self



    def predict(self, X_predict): ##预测给定的数据集，要求给定的数据集是二维矩阵
        assert self._X_train is not None and self._y_train is not None, "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1] , "col must be equaled"

        y_predict = [self._predict(x) for x in X_predict]
        return  np.array(y_predict)



    def _predict(self, x):  ##给定单个预测数据的结果值
        assert x.shape[0] == self._X_train.shape[1] , 'the feature numbers must be equealed'

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]


    def __repr__(self):
        return  "knn(k=%d)" %self.k
import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确度'''
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"

    return  sum(y_true == y_predict)/len(y_predict)


def mean_squared_error(y_true, y_predict):
    '''mse'''
    return np.sum((y_predict - y_true)**2) / len(y_true)

def root_mean_squared_error(y_ture, y_predict):
    '''rmse'''
    return sqrt(mean_squared_error(y_ture, y_predict))

def mean_absolute_error(y_true, y_predict):
    return np.sum(np.absolute(y_predict - y_true))/ len(y_true)


def r2_score(y_true, y_predict):
    '''计算y_true和y_predict之间的R—Square'''
    return 1- mean_squared_error(y_true, y_predict)/np.var(y_true)

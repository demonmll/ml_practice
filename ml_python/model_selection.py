import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据x和y按照test_ratio分割成 x_train, x_test, y_train, y_test"""

    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0 , "test_ration must be valid"

    if seed: ##测试用，如果seed为固定值，则随机数组不变
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))  ##调用permutation生成随机排序数组

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
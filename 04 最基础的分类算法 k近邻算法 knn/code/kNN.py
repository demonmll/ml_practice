import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, X_train, Y_train, x):

    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == Y_train.shape[0], "the sizes must be equaled"
    assert X_train.shape[1] == x.shape[0], "the feature numbers must be equaled"

    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train ]
    nearest = np. argsort(distances)

    topK_y = [Y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return  votes.most_common(1)[0][0]


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

def loadData(dataname):
    """
    load training and testing data from different dataset
    """
    # Monks dataset
    if dataname[:5] == 'monks':
        x_train, y_train, x_test, y_test = loadMonks(dataname)
        # normalize
        scales = np.array([[3, 3, 2, 3, 4, 2]])
        return x_train / scales, y_train, x_test / scales, y_test
    # catch error
    else:
        assert False, 'No dataset ' + dataname


def loadMonks(dataname):
    """
    load Monk problem dataset
    """
    _, index = dataname.split('-')
    assert index in ['1', '2', '3'], 'No dataset ' + dataname
    trainset = pd.read_csv('./data/monks/monks-{}.train'.format(index), header=None, delimiter=' ')
    testset = pd.read_csv('./data/monks/monks-{}.test'.format(index), header=None, delimiter=' ')
    x_train, y_train = trainset[[2,3,4,5,6,7]], trainset[1]
    x_test, y_test = testset[[2,3,4,5,6,7]], testset[1]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

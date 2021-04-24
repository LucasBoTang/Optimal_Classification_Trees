#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

def loadData(dataname):
    """
    load training and testing data from different dataset
    """
    # balance-scale
    if dataname == 'balance-scale':
        x, y = loadBalanceScale()
        return x, y
    # breast-cancer
    elif dataname == 'breast-cancer':
        x, y = loadBreastCancer()
        return x, y
    # car-evaluation
    elif dataname == 'car-evaluation':
        x, y = loadCarEvaluation()
        return x, y
    # hayes-roth
    elif dataname == 'hayes-roth':
        x, y = loadHayesRoth()
        return x, y
    # house-votes-84
    elif dataname == 'house-votes-84':
        x, y = loadHouseVotes84()
        return x, y
    # soybean-small
    elif dataname == 'soybean-small':
        x, y = loadSoybean()
        return x, y
    # spect
    elif dataname == 'spect':
        x, y = loadSpect()
        return x, y
    # tic-tac-toe
    elif dataname == 'tic-tac-toe':
        x, y = loadTicTacToe()
        return x, y
    # monks
    elif dataname[:5] == 'monks':
        x, y = loadMonks(dataname)
        return x, y
    # catch error
    else:
        raise NameError('No dataset "{}".'.format(dataname))

def oneHot(x):
    """
    one-hot encoding
    """
    x_enc = np.zeros((x.shape[0], 0))
    for j in range(x.shape[1]):
        lb = LabelBinarizer()
        lb.fit(np.unique(x[:,j]))
        x_enc = np.concatenate((x_enc, lb.transform(x[:,j])), axis=1)
    return x_enc

def loadBalanceScale():
    """
    load balance-scale dataset
    """
    df = pd.read_csv('./data/balance-scale/balance-scale.data', header=None, delimiter=',')
    x, y = df[[1,2,3,4]], df[0]
    y = pd.factorize(y)
    return np.array(x), np.array(y, dtype=object)[0]

def loadBreastCancer():
    """
    load breast-cancer dataset
    """
    df = pd.read_csv('./data/breast-cancer/breast-cancer.data', header=None, delimiter=',')
    for i in range(9):
        df = df[df[i] != '?']
    df = df.apply(lambda x: pd.factorize(x)[0])
    x, y = df[[1,2,3,4,5,6,7,8,9]], df[0]
    return np.array(x), np.array(y)

def loadCarEvaluation():
    """
    load car-evaluation dataset
    """
    df = pd.read_csv('./data/car-evaluation/car.data', header=None, delimiter=',')
    df = df.apply(lambda x: pd.factorize(x)[0])
    x, y = df[[0,1,2,3,4,5]], df[6]
    return np.array(x), np.array(y)

def loadHayesRoth():
    """
    load hayes-roth dataset
    """
    df_train = pd.read_csv('./data/hayes-roth/hayes-roth.data', header=None, delimiter=',')
    df_test = pd.read_csv('./data/hayes-roth/hayes-roth.test', header=None, delimiter=',')
    x_train, y_train = df_train[[1,2,3,4]], df_train[5]
    x_test, y_test = df_test[[0,1,2,3]], df_test[4]
    x, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    x = pd.DataFrame(x)
    x1, x2 = np.array(x[[0,3]]), np.array(x[[1,2]])
    x1 = oneHot(x1)
    x = np.concatenate((x1, x2), axis=1)
    return x, y

def loadHouseVotes84():
    """
    load house-votes-84 dataset
    """
    df = pd.read_csv('./data/house-votes-84/house-votes-84.data', header=None, delimiter=',')
    for i in range(1,17):
        df = df[df[i] != '?']
    df = df.apply(lambda x: pd.factorize(x)[0])
    x, y = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]], df[0]
    return np.array(x), np.array(y)

def loadSoybean():
    """
    load soybean dataset
    """
    df = pd.read_csv('./data/soybean-small/soybean-small.data', header=None, delimiter=',')
    for i in range(35):
        df = df[df[i] != '?']
    x, y = df[range(35)], df[35]
    y = pd.factorize(y)
    x = pd.DataFrame(x)
    x1 = np.array(x[[0,5,8,12,13,14,17,20,21,23,25,27,28,34]])
    x2 = np.array(x[[1,2,3,4,6,7,9,10,11,15,16,18,19,22,24,26,29,30,31,32,33]])
    x1 = oneHot(x1)
    x = np.concatenate((x1, x2), axis=1)
    return np.array(x), np.array(y, dtype=object)[0]

def loadSpect():
    """
    load spect dataset
    """
    df_train = pd.read_csv('./data/spect/spect.train', header=None, delimiter=',')
    df_test = pd.read_csv('./data/spect/spect.test', header=None, delimiter=',')
    x_train, y_train = df_train[range(1,23)], df_train[0]
    x_test, y_test = df_test[range(1,23)], df_test[0]
    return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)

def loadTicTacToe():
    """
    load tic-tac-toe dataset
    """
    df = pd.read_csv('./data/tic-tac-toe/tic-tac-toe.data', header=None, delimiter=',')
    x, y = df[[0,1,2,3,4,5,6,7]], df[9]
    x = oneHot(np.array(x))
    y = pd.factorize(y)
    return x, np.array(y, dtype=object)[0]

def loadMonks(dataname):
    """
    load Monks dataset
    """
    _, index = dataname.split('-')
    assert index in ['1', '2', '3'], 'No dataset ' + dataname
    df_train = pd.read_csv('./data/monks/monks-{}.train'.format(index), header=None, delimiter=' ')
    df_test = pd.read_csv('./data/monks/monks-{}.test'.format(index), header=None, delimiter=' ')
    x_train, y_train = df_train[[2,3,4,5,6,7]], df_train[1]
    x_test, y_test = df_test[[2,3,4,5,6,7]], df_test[1]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = oneHot(x)
    return x, y

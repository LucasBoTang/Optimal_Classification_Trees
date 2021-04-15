#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

import numpy as np
from scipy import stats
from gurobipy import *
import dataset

class optimalDecisionTreeClassifier:
    """
    optimal classfication tree
    """
    def __init__(self, max_depth=3, min_samples_split=2, alpha=0, timelimit=600, output=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.timelimit = timelimit
        self.output = output
        self.trained = False

    def fit(self, x, y):
        """
        fit training data
        """
        # scale data
        self.scales = np.max(x, axis=0)
        for i in range(len(self.scales)):
            if self.scales[i] == 0:
                self.scales[i] = 1

        # solve MIP
        m, a, b, c, d = self._buildMIP(x/self.scales, y)
        m.optimize()

        # get parameters
        self._a = {ind:a[ind].x for ind in a}
        self._b = {ind:b[ind].x for ind in b}
        self._c = {ind:c[ind].x for ind in c}
        self._d = {ind:d[ind].x for ind in d}

        self.trained = True

    def predict(self, x):
        """
        model prediction
        """
        assert self.trained, 'This optimalDecisionTreeClassifier instance is not fitted yet.'

        # leaf nodes
        l_index = [i for i in range(2 ** self.max_depth, 2 ** (self.max_depth + 1))]

        # leaf label
        labelmap = {}
        for t in l_index:
            for k in self.labels:
                if self._c[k,t] >= 1e-2:
                    labelmap[t] = k

        y_pred = []
        for xi in x/self.scales:
            t = 1
            while t not in l_index:
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) >= self._b[t])
                if right:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            # label
            y_pred.append(labelmap[t])

        return np.array(y_pred)


    def _buildMIP(self, x, y):
        """
        build MIP formulation for Optimal Decision Tree
        """
        # data size
        self.n, self.p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.p))

        # node index
        n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        b_index = n_index[:-2**self.max_depth] # branch nodes
        l_index = n_index[-2**self.max_depth:] # leaf nodes
        # labels
        self.labels = np.unique(y)

        # create a model
        m = Model('m')
       
        # time limit
        m.Params.timelimit = self.timelimit
        # output
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        # parallel
        m.params.threads = 0

        # model sense
        m.modelSense = GRB.MINIMIZE

        # varibles
        a = m.addVars(self.p, b_index, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(b_index, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        c = m.addVars(self.labels, l_index, vtype=GRB.BINARY, name='c') # node prediction
        d = m.addVars(b_index, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        L = m.addVars(l_index, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassfication
        M = m.addVars(self.labels, l_index, vtype=GRB.CONTINUOUS, name='M') # leaf node samples with label
        N = m.addVars(l_index, vtype=GRB.CONTINUOUS, name='N') # leaf node samples

        # calculate baseline accuracy
        baseline = self._calBaseline(y)

        # calculate minimum distance
        min_dis = self._calMinDist(x)

        # objective function
        obj = L.sum() / baseline + self.alpha * d.sum()
        m.setObjective(obj)

        # constraints
        # (20)
        m.addConstrs(L[t] >= N[t] - M[k,t] - self.n * (1 - c[k,t]) for t in l_index for k in self.labels)
        # (21)
        m.addConstrs(L[t] <= N[t] - M[k,t] + self.n * c[k,t] for t in l_index for k in self.labels)
        # (17)
        m.addConstrs(quicksum(((1 if y[i] == k else -1) + 1) * z[i,t] for i in range(self.n)) / 2 == M[k,t]
                     for t in l_index for k in self.labels)
        # (16)
        m.addConstrs(z.sum('*', t) == N[t] for t in l_index)
        # (18)
        m.addConstrs(c.sum('*', t) == l[t] for t in l_index)
        # (13) and (14)
        for t in l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(quicksum(a[j,ta] * (x[i,j] + min_dis[j]) for j in range(self.p))
                                 +
                                 (1 + np.max(min_dis)) * (1 - d[ta])
                                 <=
                                 b[ta] + (1 + np.max(min_dis)) * (1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(quicksum(a[j,ta] * x[i,j] for j in range(self.p))
                                 >=
                                 b[ta] - (1 - z[i,t])
                                 for i in range(self.n))
                left = (ta % 2 == 0)
                ta //= 2
        # (8)
        m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))
        # (6)
        m.addConstrs(z[i,t] <= l[t] for t in l_index for i in range(self.n))
        # (7)
        m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in l_index)
        # (2)
        m.addConstrs(a.sum('*', t) == d[t] for t in b_index)
        # (3)
        m.addConstrs(b[t] <= d[t] for t in b_index)
        # (5)
        m.addConstrs(d[t] <= d[t//2] for t in b_index if t != 1)

        return m, a, b, c, d

    def _calBaseline(self, y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    def _calMinDist(self, x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

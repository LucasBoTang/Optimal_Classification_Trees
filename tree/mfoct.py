#!/usr/bin/env python
# coding: utf-8
# author: Bo Lin

from collections import namedtuple
import numpy as np
import gurobipy as gp
from scipy import stats
from sklearn import tree

class maxFlowOptimalDecisionTreeClassifier:
    def __init__(self, max_depth=3, alpha=0, warmstart=True, timelimit=600, output=True):
        self.max_depth = max_depth
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.trained = False
        self.optgap = None

        # initialize the tree
        self.B = list(range(2 ** self.max_depth - 1))
        self.T = list(range(2 ** self.max_depth - 1, 2 ** (self.max_depth + 1) - 1))
        self.A = [self._tree_ancestor(n) for n in self.B + self.T]

    def fit(self, x, y):
        """
        fit training data
        """
        # data size
        self.n, self.m = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.m))

        # the number of distinct labels
        self.K = list(range(np.max(y)+1))
        # the set of training data
        self.I = list(range(self.n))
        # the set of features
        self.F = list(range(self.m))

        # solve MIP
        m, b, p, w = self._buildMIP(x, y)
        if self.warmstart:
            self._setStart(x, y, b, p, w)
        m.optimize(self._min_cut)
        self.optgap = m.MIPGap

        self._tree_construction(m, b, w)
        self.trained = True

    def _buildMIP(self, x, y):
        """
        build MIP formulation for Max-Flow Optimal Decision Tree
        """
        # initialize the master problem
        m = gp.Model('m')
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        m.Params.timelimit = self.timelimit
        # parallel
        m.params.threads = 0

        # add decision variables
        b = m.addVars(self.B, self.F, name='b', vtype=gp.GRB.BINARY)
        w = m.addVars(self.B + self.T, self.K, name='w', vtype=gp.GRB.BINARY)
        g = m.addVars(self.I, name='g', lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)
        p = m.addVars(self.B + self.T, name='p', vtype=gp.GRB.BINARY)

        # add constraints
        m.addConstrs((gp.quicksum(b[n, f] for f in self.F)
                      + p[n]
                      + gp.quicksum(p[m] for m in self.A[n])
                      == 1
                      for n in self.B),
                      name='branching_nodes')

        m.addConstrs((p[n]
                      + gp.quicksum(p[m] for m in self.A[n])
                      == 1
                      for n in self.T),
                      name='terminal_nodes')

        m.addConstrs((gp.quicksum(w[n, k] for k in self.K) == p[n]
                                  for n in self.B + self.T),
                      name='label_assignment')

        # set objective function
        baseline = self._calBaseline(y)
        obj = 1 / baseline * g.sum() - self.alpha * b.sum()
        m.setObjective(obj, gp.GRB.MAXIMIZE)

        # enable lazy cuts
        m.Params.lazyConstraints = 1
        m._I = self.I
        m._B = self.B
        m._T = self.T
        m._F = self.F
        m._X = x
        m._Y = y
        m._K = self.K

        m._b = b
        m._w = w
        m._p = p
        m._g = g

        return m, b, p, w

    def stable_fit_robust(self, x, y, N):
        """
        fit training data with min-max method
        """
        # data size
        self.n, self.m = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.m))

        # the number of distinct labels
        self.K = list(range(np.max(y) + 1))
        # the set of training data
        self.I = list(range(self.n))
        # the set of features
        self.F = list(range(self.m))

        # solve MIP
        m, b, w = self._buildMIP_robust(x, y, N)
        m.optimize(self._stable_robust_min_cut)
        self.optgap = m.MIPGap

        self._tree_construction(m, b, w)
        self.trained = True

    def _buildMIP_robust(self, x, y, N):
        """
        build MIP formulation for Max-Flow Optimal Decision Tree with min-max method
        """
        # initialize the master problem
        m = gp.Model('m')
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        m.Params.timelimit = self.timelimit
        # parallel
        m.params.threads = 0

        # add decision variables
        b = m.addVars(self.B, self.F, name='b', vtype=gp.GRB.BINARY)
        w = m.addVars(self.B + self.T, self.K, name='w', vtype=gp.GRB.BINARY)
        p = m.addVars(self.B + self.T, name='p', vtype=gp.GRB.BINARY)
        u = m.addVars(self.I, name='u', vtype=gp.GRB.CONTINUOUS)
        theta = m.addVar(name='theta', lb=-gp.GRB.INFINITY, ub=len(y)*1000, vtype=gp.GRB.CONTINUOUS)

        # add constraints
        m.addConstrs((gp.quicksum(b[n, f] for f in self.F)
                      + p[n]
                      + gp.quicksum(p[m] for m in self.A[n])
                      == 1
                      for n in self.B),
                      name='branching_nodes')

        m.addConstrs((p[n]
                     + gp.quicksum(p[m] for m in self.A[n])
                     == 1
                     for n in self.T),
                     name='terminal_nodes')

        m.addConstrs((gp.quicksum(w[n, k] for k in self.K) == p[n]
                                  for n in self.B + self.T),
                     name='label_assignment')

        # set objective function
        baseline = self._calBaseline(y)
        obj = 1 / baseline * (N * theta - u.sum()) - \
              self.alpha * b.sum()
        m.setObjective(obj, gp.GRB.MAXIMIZE)

        # enable lazy cuts
        m.Params.lazyConstraints = 1
        m._I = self.I
        m._B = self.B
        m._T = self.T
        m._F = self.F
        m._X = x
        m._Y = y
        m._K = self.K

        m._b = b
        m._w = w
        m._p = p
        m._u = u
        m._theta = theta

        return m, b, w

    def stable_fit_CP(self, x, y, N):
        """
        fit training data with cutting plane
        """
        # data size
        self.n, self.m = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.m))

        # the number of distinct labels
        self.K = list(range(np.max(y) + 1))
        # the set of training data
        self.I = list(range(self.n))
        # the set of features
        self.F = list(range(self.m))

        # solve MIP
        m, b, w = self._buildMIP_CP(x, y, N)
        m.optimize(self._stable_CP_min_cut)
        self.optgap = m.MIPGap

        self._tree_construction(m, b, w)
        self.trained = True

    def _buildMIP_CP(self, x, y, N):
        """
        build MIP formulation for Max-Flow Optimal Decision Tree with min-max method
        """
        # initialize the master problem
        m = gp.Model('m')
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        m.Params.timelimit = self.timelimit
        # parallel
        m.params.threads = 0

        # add decision variables
        b = m.addVars(self.B, self.F, name='b', vtype=gp.GRB.BINARY)
        w = m.addVars(self.B + self.T, self.K, name='w', vtype=gp.GRB.BINARY)
        g = m.addVars(self.I, name='g', lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)
        p = m.addVars(self.B + self.T, name='P', vtype=gp.GRB.BINARY)
        t = m.addVar(name = 't', vtype=gp.GRB.CONTINUOUS, ub = len(y)+1, lb = 0)

        # add constraints
        m.addConstrs((gp.quicksum(b[n, f] for f in self.F)
                      + p[n]
                      + gp.quicksum(p[m] for m in self.A[n])
                      == 1
                      for n in self.B),
                     name='branching_nodes')

        m.addConstrs((p[n]
                      + gp.quicksum(p[m] for m in self.A[n])
                      == 1
                      for n in self.T),
                     name='terminal_nodes')

        m.addConstrs((gp.quicksum(w[n, k] for k in self.K) == p[n]
                                  for n in self.B + self.T),
                     name='label_assignment')

        # set objective function
        baseline = self._calBaseline(y)
        obj = 1 / baseline * t - self.alpha * b.sum()
        m.setObjective(obj, gp.GRB.MAXIMIZE)

        # enable lazy cuts
        m.Params.lazyConstraints = 1
        m._I = self.I
        m._B = self.B
        m._T = self.T
        m._F = self.F
        m._X = x
        m._Y = y
        m._K = self.K
        m._N = N

        m._b = b
        m._w = w
        m._p = p
        m._g = g
        m._t = t

        return m, b, w

    def predict(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This binOptimalDecisionTreeClassifier instance is not fitted yet.')

        pred = []
        for val in x:
            n = 0
            while n not in self.labels:
                f = self.branches[n]
                if val[f] == 0:
                    n = n * 2 + 1
                else:
                    n = n * 2 + 2
            pred.append(self.labels[n])
        return np.array(pred)

    @staticmethod
    def _tree_children(node):
        """
        get left and right children of node in the tree
        """
        return 2 * node + 1, 2 * node + 2

    @staticmethod
    def _tree_parent(node):
        """
        get parent of node in the tree
        """
        if node == 0:
            return None
        if node % 2 == 1:
            return int(node // 2)
        return int(node // 2) - 1

    def _tree_ancestor(self, node):
        """
        find the accessor of the given node in the tree
        """
        ancestor = []
        parent = self._tree_parent(node)

        while parent is not None:
            ancestor.append(parent)
            parent = self._tree_parent(parent)

        return ancestor

    def _tree_construction(self, m, b, w):
        """
        construct the decision tree
        """
        b_val = m.getAttr('x', b)
        w_val = m.getAttr('x', w)

        self.branches = {n: f for n in self.B for f in self.F if b_val[n,f] >= 0.999}
        self.labels = {n: k for n in self.B + self.T for k in self.K if w_val[n, k] >= 0.999}

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _min_cut(model, where):
        """
        lazy constraints
        """
        if where == gp.GRB.Callback.MIPSOL:
            b_val = model.cbGetSolution(model._b)
            w_val = model.cbGetSolution(model._w)
            p_val = model.cbGetSolution(model._p)
            g_val = model.cbGetSolution(model._g)

            for i in model._I:
                S, branch = [], []
                n = 0

                while (p_val[n] <= 0.9):
                    S.append(n)
                    if np.sum([b_val[n, f] for f in model._F if model._X[i, f] <= 1e-5]) >= 0.999:
                        n = 2 * n + 1
                        branch.append(0)
                    elif np.sum([b_val[n, f] for f in model._F if model._X[i, f] >= 0.999]) >= 0.999:
                        n = 2 * n + 2
                        branch.append(1)
                    else:
                        raise ValueError

                S.append(n)

                if g_val[i] > np.sum([w_val[n, k] for k in model._K if model._Y[i] == k]) + 0.1:
                    if n in model._B:
                        model.cbLazy(model._g[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i,f] == 1 - branch[idx]) +
                                     gp.quicksum(model._b[n, f] for f in model._F) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )
                    else:
                        model.cbLazy(model._g[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i, f] == 1 - branch[idx]) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )

    @staticmethod
    def _stable_robust_min_cut(model, where):
        """
        lazy constraints
        """
        if where == gp.GRB.Callback.MIPSOL:
            b_val = model.cbGetSolution(model._b)
            w_val = model.cbGetSolution(model._w)
            p_val = model.cbGetSolution(model._p)
            theta_val = model.cbGetSolution(model._theta)
            u_val = model.cbGetSolution(model._u)

            for i in model._I:
                S, branch = [], []
                n = 0

                while (n in model._B) and (p_val[n] == 0):
                    S.append(n)
                    if np.sum([b_val[n, f] for f in model._F if model._X[i, f] <= 1e-5]) >= 0.999:
                        n = 2 * n + 1
                        branch.append(0)
                    elif np.sum([b_val[n, f] for f in model._F if model._X[i, f] >= 0.999]) >= 0.999:
                        n = 2 * n + 2
                        branch.append(1)

                S.append(n)

                # print(theta_val, u_val[i], np.sum([w_val[n, k] for k in model._K if model._Y[i] == k]))
                if theta_val - u_val[i] > np.sum([w_val[n, k] for k in model._K if model._Y[i] == k]) + 0.1:
                    if n in model._B:
                        model.cbLazy(model._theta - model._u[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i, f] == 1 - branch[idx]) +
                                     gp.quicksum(model._b[n, f] for f in model._F) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )
                    else:
                        model.cbLazy(model._theta - model._u[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i, f] == 1 - branch[idx]) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )

    @staticmethod
    def _stable_CP_min_cut(model, where):
        """
        lazy constraints
        """
        if where == gp.GRB.Callback.MIPSOL:
            b_val = model.cbGetSolution(model._b)
            w_val = model.cbGetSolution(model._w)
            p_val = model.cbGetSolution(model._p)
            g_val = model.cbGetSolution(model._g)
            t_val = model.cbGetSolution(model._t)

            miss = []
            arr = []
            counter = 0

            for i in model._I:
                S, branch = [], []
                n = 0

                while (p_val[n] <= 0.9):
                    S.append(n)
                    if np.sum([b_val[n, f] for f in model._F if model._X[i, f] <= 1e-5]) >= 0.999:
                        n = 2 * n + 1
                        branch.append(0)
                    elif np.sum([b_val[n, f] for f in model._F if model._X[i, f] >= 0.999]) >= 0.999:
                        n = 2 * n + 2
                        branch.append(1)
                    else:
                        raise ValueError('Features should be binary')
                S.append(n)

                if np.sum([w_val[n, k] for k in model._K if model._Y[i] == k]) <= 0.1:
                    miss.append(i)
                else:
                    arr.append(i)

                if g_val[i] > np.sum([w_val[n, k] for k in model._K if model._Y[i] == k]) + 0.1:
                    counter += 1
                    if n in model._B:
                        model.cbLazy(model._g[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i,f] == 1 - branch[idx]) +
                                     gp.quicksum(model._b[n, f] for f in model._F) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )
                    else:
                        model.cbLazy(model._g[i] <=
                                     gp.quicksum(model._b[val, f]
                                                 for idx, val in enumerate(S[:-1])
                                                 for f in model._F
                                                 if model._X[i, f] == 1 - branch[idx]) +
                                     gp.quicksum(model._w[j, k]
                                                 for j in S for k in model._K
                                                 if model._Y[i] == k)
                                     )

            rhs = model._N - len(miss)
            if t_val > rhs and counter == 0:
                if rhs > 0:
                    choice = np.random.choice(arr, rhs, replace=False)
                    model.cbLazy(
                        model._t <= gp.quicksum(model._g[i] for i in choice) + gp.quicksum(model._g[i] for i in miss))
                else:
                    choice = np.random.choice(miss, model._N, replace=False)
                    model.cbLazy(model._t <= gp.quicksum(model._g[i] for i in choice))

    def _setStart(self, x, y, b, p, w):
        """
        set warm start from CART
        """
        # train with CART
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(x, y)

        # get splitting rules
        rules = self._getRules(clf)

        # fix branch node
        for n in self.B:
            # split
            if rules[n].feat is not None:
                for f in self.F:
                    if f == int(rules[n].feat):
                        b[n, f].start = 1
                    else:
                        b[n, f].start = 0
            # not split
            else:
                for f in self.F:
                    b[n, f].start = 0

        # fix terminal nodes
        # branch node
        for n in self.B:
            # terminate
            if rules[n].feat == tree._tree.TREE_UNDEFINED:
                p[n].start = 1
                for k in self.K:
                    if k == np.argmax(rules[n].value):
                        w[n, k].start = 1
                    else:
                        w[n, k].start = 0
            # not terminate
            else:
                p[n].start = 0
                for k in self.K:
                    w[n, k].start = 0
        # leaf node
        for n in self.T:
            # pruned
            if rules[n].value is None:
                p[n].start = 0
                for k in self.K:
                    w[n, k].start = 0
            # not pruned
            else:
                p[n].start = 1
                for k in self.K:
                    if k == np.argmax(rules[n].value):
                        w[n, k].start = 1
                    else:
                        w[n, k].start = 0

    def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {0:0}
        for n in self.B:
            # terminal
            node_map[2*n+1] = -1
            node_map[2*n+2] = -1
            # left
            l = clf.tree_.children_left[node_map[n]]
            node_map[2*n+1] = l
            # right
            r = clf.tree_.children_right[node_map[n]]
            node_map[2*n+2] = r

        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for n in self.B:
            i = node_map[n]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[n] = r
        # leaf nodes
        for n in self.T:
            i = node_map[n]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[n] = r

        return rules

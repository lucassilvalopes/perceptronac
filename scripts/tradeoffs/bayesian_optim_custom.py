"""
https://machinelearningmastery.com/what-is-bayesian-optimization/

https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
"""


import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter


class BOCustom:
	
    def __init__(self,f,pbounds,**kwargs):
        self.f = f
        self.pbounds = pbounds
        self.model = GaussianProcessRegressor()
        self.exploration_exploitation_tredeoff = 1
        self.max = None
        self.res = []

    def surrogate(self, X):
        with catch_warnings():
            simplefilter("ignore")
            return self.model.predict(X, return_std=True)

    def acquisition(self, Xsamples):
        # upper confidence bound acquisition function
        mu, std = self.surrogate(Xsamples)
        # mu = mu[:, 0]
        a = mu + self.exploration_exploitation_tredeoff * std
        return a

    def random(self,n_samples):
        features = []
        for i,k in enumerate(self.f.__code__.co_varnames):
            features.append(
                [random.uniform(self.pbounds[k][0], self.pbounds[k][1]) for _ in range(n_samples)]
            )
        Xsamples = np.array(features).T
        return Xsamples

    def opt_acquisition(self):
        Xsamples = self.random(100)
        scores = self.acquisition(Xsamples)
        ix = np.argmax(scores)
        x = Xsamples[ix, :]
        actual = self.f(*x)
        self.res.append(self.format_res(x, actual))
        return x
    
    def format_res(self,x,y):
        return {"params": dict(zip(self.f.__code__.co_varnames,x)), "target": y}

    def maximize(self,init_points=5,n_iter=25):
        X = self.random(init_points)
        y = np.asarray([self.f(*x) for x in X]).reshape(len(y), 1)
        for i in range(len(y)):
            self.res.append(self.format_res(X[i, :],y[i]))
        self.model.fit(X, y)
        for i in range(n_iter):
            x = self.opt_acquisition()
            actual = self.f(*x)
            X = np.vstack((X, [x]))
            y = np.vstack((y, [[actual]]))
            self.model.fit(X, y)
        ix = np.argmax(y)
        self.max= self.format_res(X[ix], y[ix])


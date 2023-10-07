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
	
    def __init__(self,f,pbounds):
        self.f = f
        self.pbounds = pbounds
        self.model = GaussianProcessRegressor()
        self.exploration_exploitation_tredeoff = 1

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
        return Xsamples[ix, :]

    def maximize(self,init_points=5,n_iter=25):
        X = self.random(init_points)
        y = np.asarray([self.f(*x) for x in X]).reshape(len(y), 1)
        self.model.fit(X, y)
        for i in range(n_iter):
            x = self.opt_acquisition()
            actual = self.f(*x)
            X = np.vstack((X, [x]))
            y = np.vstack((y, [[actual]]))
            self.model.fit(X, y)
        ix = np.argmax(y)
        return {**dict(zip(self.f.__code__.co_varnames,X[ix])), "target": y[ix]}


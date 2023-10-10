"""
https://machinelearningmastery.com/what-is-bayesian-optimization/

https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
"""


import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm


class BOCustom:
	
    def __init__(self,f,pbounds,verbose=None,random_state=None,lambda_grid=None):
        self.f = f
        self.pbounds = pbounds
        if lambda_grid is None:
            self.lambda_grid = np.ones((1,len(pbounds)))
        else:
            self.lambda_grid = lambda_grid
        self.models = self.init_models()
        self.max = None
        self.res = []
        if random_state is not None:
            random.seed(random_state)
    
    def init_models(self):
        return [GaussianProcessRegressor() for _ in range(self.lambda_grid.shape[0])]

    def surrogate(self, model, X):
        with catch_warnings():
            simplefilter("ignore")
            return model.predict(X, return_std=True)

    def pi_acquisition(self, X, Xsamples):
        # probability of improvement acquisition function
        probs = []
        for model in self.models:
            yhat, _ = self.surrogate(model,X)
            best = max(yhat)
            mu, std = self.surrogate(model,Xsamples)
            mu = mu[:, 0]
            probs.append( norm.cdf((mu - best) / (std+1E-9)) )
        return sum(probs)

    def list_func_args(self):
        return [
            a for a in self.f.__code__.co_varnames[:self.f.__code__.co_argcount] if a !="self"]

    def random(self,n_samples):
        features = []
        for i,k in enumerate(self.list_func_args()):
            features.append(
                [random.uniform(self.pbounds[k][0], self.pbounds[k][1]) for _ in range(n_samples)]
            )
        Xsamples = np.array(features).T
        return Xsamples

    def opt_acquisition(self,X):
        Xsamples = self.random(100)
        scores = self.pi_acquisition(X,Xsamples)
        ix = np.argmax(scores)
        x = Xsamples[ix, :]
        for m in range(len(self.models)):
            actual = self.f(*list(zip(x,self.lambda_grid[m])))
            self.res.append(self.format_res(x, actual, m))
        return x
    
    def format_res(self,x,y,m):
        return {"params": dict(zip(self.list_func_args(),x)), "target": y, "model": m}

    def maximize(self,init_points=5,n_iter=25):
        X = self.random(init_points)
        ys = []
        for m in range(len(self.models)):
            y = np.asarray([self.f(*list(zip(x,self.lambda_grid[m]))) for x in X])
            y = y.reshape(len(y), 1)
            self.models[m].fit(X, y)
            ys.append(y)
        for i in range(init_points):
            for m in range(len(self.models)):
                self.res.append(self.format_res(X[i, :],ys[m][i],m))
        for i in range(n_iter):
            x = self.opt_acquisition(X)
            X = np.vstack((X, [x]))
            for m in range(len(self.models)):
                actual = self.f(*list(zip(x,self.lambda_grid[m])))
                ys[m] = np.vstack((ys[m], [[actual]]))
                self.models[m].fit(X, ys[m])
        self.maxes = []
        for m in range(len(self.models)):
            ix = np.argmax(ys[m])
            self.maxes.append( self.format_res(X[ix], ys[m][ix],m) )
        self.max = self.maxes[0] # use if len(self.models) == 1, otherwise use maxes


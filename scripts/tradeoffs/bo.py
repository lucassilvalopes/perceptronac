"""
https://machinelearningmastery.com/what-is-bayesian-optimization/

https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html

https://www.w3schools.com/python/ref_random_setstate.asp

https://stackoverflow.com/questions/45922944/what-is-the-exact-nature-of-differences-or-similarities-between-random-setstate
"""


import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

class BOCustom:
	
    def __init__(self,f,pbounds,verbose=None,random_state=None,lambda_grid=None,acquisition_func="pii"):
        """
        acquisition_func: 
        - probability of improvement (pi)
        - probability of improvement independent (pii)
        - random
        """
        self.f = f
        self.pbounds = pbounds
        if lambda_grid is None:
            self.lambda_grid = np.ones((1,len(self.list_func_args())))
        else:
            self.lambda_grid = lambda_grid
        self.models = self.init_models()
        self.max = None
        self.res = []
        self.original_random_state = random.getstate()
        random.seed(random_state)
        self.acquisition_func = acquisition_func
    
    def __del__(self):
        random.setstate(self.original_random_state)

    def init_models(self):
        kernel = DotProduct() + WhiteKernel()
        return [GaussianProcessRegressor(kernel=kernel) for _ in range(self.lambda_grid.shape[0])]

    def surrogate(self, model, X):
        with catch_warnings():
            simplefilter("ignore")
            return model.predict(X, return_std=True)

    def pi_acquisition(self, X, Xsamples):
        # multidimensional probability of improvement acquisition function
        # with correlation between planes modeled by the normal vectors inner product 
        means = []
        stds = []
        bests = []
        for model in self.models:
            yhat, _ = self.surrogate(model,X)
            best = np.max(yhat)
            mu, std = self.surrogate(model,Xsamples)
            mu = mu[:, 0]
            means.append( mu )
            stds.append( std )
            bests.append( best )

        means = np.array(means).T # (n_samples, n_planes)
        stds = np.array(stds).T # (n_samples, n_planes)
        bests = np.array(bests) # (n_planes,)

        n_samples = Xsamples.shape[0]

        norms = np.sqrt(np.sum(self.lambda_grid**2,axis=1)).reshape(-1,1)
        corr = (self.lambda_grid @ self.lambda_grid.T) / (norms @ norms.T)

        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # import matplotlib.pyplot as plt
        # plt.imshow(corr)
        # plt.show(block=True)

        probs = []
        for i in range(n_samples):

            mean_vec = means[i] # (n_planes,)
            std_vec = stds[i] # (n_planes,)
            best_vec = bests # (n_planes,)

            # https://en.wikipedia.org/wiki/Covariance_matrix#Relation_to_the_correlation_matrix
            covariance_mtx = np.diag(std_vec) @ corr @ np.diag(std_vec)

            assert np.allclose(std_vec*std_vec,np.diag(covariance_mtx))

            dist = mvn(mean=mean_vec, cov=covariance_mtx, allow_singular=True)

            prob = 1 - dist.cdf(best_vec)

            probs.append(prob)
        
        return np.array(probs)

    def pi_acquisition_independent(self, X, Xsamples):
        # multidimensional probability of improvement acquisition function
        # assuming independence between planes
        probs = []
        for model in self.models:
            yhat, _ = self.surrogate(model,X)
            best = max(yhat)
            mu, std = self.surrogate(model,Xsamples)
            mu = mu[:, 0]
            probs.append( norm.cdf((best - mu) / (std+1E-9)) )
        probs = np.vstack(probs)
        return 1 - np.prod(probs,axis=0)

    def pi_acquisition_random(self, X, Xsamples):
        return np.ones((Xsamples.shape[0],))

    def list_func_args(self):
        """
        https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
        https://stackoverflow.com/questions/57885922/count-positional-arguments-in-function-signature
        """
        n_all_args = self.f.__code__.co_argcount
        n_kwargs = 0 if self.f.__defaults__ is None else len(self.f.__defaults__)
        n_args = n_all_args - n_kwargs
        return [a for a in self.f.__code__.co_varnames[:n_args] if a !="self"]

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
        if self.acquisition_func == "pii":
            scores = self.pi_acquisition_independent(X,Xsamples)
        elif self.acquisition_func == "pi":
            scores = self.pi_acquisition(X,Xsamples)
        elif self.acquisition_func == "random":
            scores = self.pi_acquisition_random(X,Xsamples)
        else:
            raise ValueError(self.acquisition_func)
        # print(scores)
        ix = np.argmax(scores)
        x = Xsamples[ix, :]
        for m in range(len(self.models)):
            actual = self.f(*x,dynamic_weights=self.lambda_grid[m])
            self.res.append(self.format_res(x, actual, m))
        return x
    
    def format_res(self,x,y,m):
        return {"params": dict(zip(self.list_func_args(),x)), "target": y, "weights": self.lambda_grid[m]}

    def maximize(self,init_points=5,n_iter=25):
        X = self.random(init_points)
        ys = []
        for m in range(len(self.models)):
            y = np.asarray([self.f(*x,dynamic_weights=self.lambda_grid[m]) for x in X])
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
                actual = self.f(*x,dynamic_weights=self.lambda_grid[m])
                ys[m] = np.vstack((ys[m], [[actual]]))
                self.models[m].fit(X, ys[m])
        self.maxes = []
        for m in range(len(self.models)):
            ix = np.argmax(ys[m])
            self.maxes.append( self.format_res(X[ix], ys[m][ix],m) )
        self.max = self.maxes[0] # use if len(self.models) == 1, otherwise use maxes


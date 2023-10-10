"""
https://github.com/bayesian-optimization/BayesianOptimization
https://stackoverflow.com/questions/57182358/iterate-over-integers-in-bayesian-optimization-package-python
https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
"""

import pandas as pd
import numpy as np
# from bayes_opt import BayesianOptimization
from bayesian_optim_custom import BOCustom as BayesianOptimization
from collections.abc import Iterable

class BayesOptRateDist:

    def __init__(self,csv_path,axes,weights,lambdas=[]):

        self.data = self.read_data(csv_path,lambdas)
        self.axes = axes
        self.weights = weights
        self.possible_values = {
            "D": [3,4],
            "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
            "N": [32, 64, 96, 128, 160, 192, 224],
            "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        }
        self.pbounds = {k:(float(v[0]),float(v[-1])) for k,v in self.possible_values.items()}

    @staticmethod
    def read_data(csv_path,lambdas):
        data = pd.read_csv(csv_path)
        if len(lambdas) > 0:
            data = data[data["labels"].apply(lambda x: any([(lmbd in x) for lmbd in lambdas]) )]
        data = data.set_index("labels")
        return data

    @staticmethod
    def to_str_method(D,L,N,M):
        return f"D{D}L{L}N{N}M{M}"

    def get_label_coord(self,label):
        return self.data.loc[label,self.axes].values.tolist()

    @staticmethod
    def unpack_black_box_function_inputs(D,L,N,M):
        inputs_is_iter = [(not isinstance(P,str)) and isinstance(P,Iterable) for P in [D,L,N,M]]
        if all(inputs_is_iter):
            D,Dw = D[0],D[1]
            L,Lw = L[0],L[1]
            N,Nw = N[0],N[1]
            M,Mw = M[0],M[1]
        elif any(inputs_is_iter):
            raise ValueError("inconsistent input types. Either all or none should be iterable.")
        else:
            Dw,Lw,Nw,Mw = 1,1,1,1
        return D,L,N,M,Dw,Lw,Nw,Mw

    def black_box_function(self,D,L,N,M):
        """receives hyperparameters and outputs -J = -(R + lambda * D + gamma * C)"""

        D,L,N,M,Dw,Lw,Nw,Mw = self.unpack_black_box_function_inputs(D,L,N,M)

        D,L,N,M = self.round_to_possible_values(D,L,N,M)

        coord = self.get_label_coord(self.to_str_method(D,L,N,M))
        J = sum([c*dynamic_w*fixed_w for c,dynamic_w,fixed_w in zip(coord,[Dw,Lw,Nw,Mw],self.weights)])
        return -J

    def round_to_possible_values(self,D,L,N,M):
    
        D = self.round_param_to_possible_values(D,self.possible_values["D"])
        L = self.round_param_to_possible_values(L,self.possible_values["L"])
        N = self.round_param_to_possible_values(N,self.possible_values["N"])
        M = self.round_param_to_possible_values(M,self.possible_values["M"])
    
        return D,L,N,M

    def round_param_to_possible_values(self,P,P_list):
        if P >= float(P_list[-1]):
            return P_list[-1]
        if P < float(P_list[0]):
            return P_list[0]
        for i,lb,ub in zip(range(len(P_list)-1),P_list[:-1],P_list[1:]):
            if float(lb) <= P < float(ub):
                percent = (P - float(lb))/(float(ub) - float(lb))
                if percent < 0.5:
                    return P_list[i]
                else:
                    return P_list[i+1]


def simple_lambda_grid():
    m45 = -45
    m90 = -90+1e-10
    m00 = -1e-10
    grid = -1/np.tan((np.pi/180) * np.array([
        [m45,m90,m90],
        [m45,m90,m45],
        [m45,m90,m00],
        [m45,m45,m90],
        [m45,m45,m45],
        [m45,m00,m90],
        [m45,m00,m00]
    ]))
    return grid



def bayes_opt_rate_dist(csv_path,axes,weights,lambdas=[],random_state=1,init_points=5,n_iter=25):

    bayesOptRateDist = BayesOptRateDist(csv_path,axes,weights,lambdas=lambdas)


    optimizer = BayesianOptimization(
        f=bayesOptRateDist.black_box_function,
        pbounds=bayesOptRateDist.pbounds,
        verbose=2,
        random_state=random_state,
    )
    # optimizer.set_gp_params(alpha=1e-3)


    # n_iter: How many steps of bayesian optimization you want to perform.
    # init_points: How many steps of random exploration you want to perform.
    # total iterations = n_iter + init_points
    optimizer.maximize(
        init_points=init_points, # default: 5
        n_iter=n_iter, # default: 25
    )

    # best combination of parameters and target value found
    print(optimizer.max)

    # list of all parameters probed and their corresponding target values 
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    lbl = bayesOptRateDist.to_str_method(
        *bayesOptRateDist.round_to_possible_values(
            optimizer.max["params"]["D"],
            optimizer.max["params"]["L"],
            optimizer.max["params"]["N"],
            optimizer.max["params"]["M"]
        )
    )

    print(
        lbl, bayesOptRateDist.get_label_coord(lbl), optimizer.max["target"]
    )

if __name__ == "__main__":

    bayes_opt_rate_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss"],
        [1,2e-2*(255**2)],
        lambdas=["2e-2"]
    )

    bayes_opt_rate_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        [1,2e-2*(255**2),1/1e6],
        lambdas=["2e-2"]
    )

    bayes_opt_rate_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","flops"],
        [1,2e-2*(255**2),1/1e10],
        lambdas=["2e-2"]
    )
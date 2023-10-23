"""
https://github.com/bayesian-optimization/BayesianOptimization
https://stackoverflow.com/questions/57182358/iterate-over-integers-in-bayesian-optimization-package-python
https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
"""

import random
import numpy as np
import pandas as pd
# from bayes_opt import BayesianOptimization
from bo import BOCustom as BayesianOptimization
from collections.abc import Iterable

class BayesOptRateDist:

    def __init__(self,csv_path,axes,weights,lambdas=[]):

        self.axes = axes
        self.weights = weights
        self.possible_values = {
            "D": [3,4],
            "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
            "N": [32, 64, 96, 128, 160, 192, 224],
            "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        }
        self.pbounds = {k:(float(v[0]),float(v[-1])) for k,v in self.possible_values.items()}
        self.data = self.read_data(csv_path,lambdas)

    def read_data(self,csv_path,lambdas):
        if csv_path == "random":
            labels = [
                self.to_str_method(D,L,N,M)
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]
            params = [
                # fake function based if it was an MLP
                # without bias and with input size 10000 = (100 * 100)
                N*10000 + (N ** 2) * (D-1) + (N ** 2) * (D-2) + 2*N*M 
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]
            maxN = self.pbounds["N"][1]
            maxM = self.pbounds["M"][1]
            maxD = self.pbounds["D"][1]
            N0bpp = 192
            M0bpp = 256
            D0bpp = 3
            N0mse = 160
            M0mse = 288
            D0mse = 4
            widthNbpp = 1.1
            widthMbpp = 0.9
            widthDbpp = 1
            widthNmse = 1.2
            widthMmse = 1.0
            widthDmse = 1.3
            bpp_loss = [(
                    ((N/maxN - N0bpp/maxN)**2)/widthNbpp
                    + ((M/maxM - M0bpp/maxM)**2)/widthMbpp
                    + ((D/maxD - D0bpp/maxD)**2)/widthDbpp
                    + max(min(np.log10(float(L)),8),-8) + 8.5
                )
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]
            mse_loss = [(
                    ((N/maxN - N0mse/maxN)**2)/widthNmse
                    + ((M/maxM - M0mse/maxM)**2)/widthMmse
                    + ((D/maxD - D0mse/maxD)**2)/widthDmse
                    - max(min(np.log10(float(L)),8),-8) + 8.5
                )
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]
            data = pd.DataFrame({
                "labels": labels,
                "bpp_loss": bpp_loss,
                "mse_loss": mse_loss,
                "params": params
            })
        else:

            data = pd.read_csv(csv_path)
            if len(lambdas) > 0:
                data = data[data["labels"].apply(lambda x: any([(lmbd in x) for lmbd in lambdas]) )]
            data["params"] = data["params"]/1e+6

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data[self.axes] = scaler.fit_transform(data[self.axes])

        data = data.set_index("labels")
        return data

    @staticmethod
    def to_str_method(D,L,N,M):
        return f"D{D}L{L}N{N}M{M}"

    def get_label_coord(self,label):
        return self.data.loc[label,self.axes].values.tolist()

    def black_box_function(self,D,L,N,M,dynamic_weights=None):
        """receives hyperparameters and outputs -J = -(R + lambda * D + gamma * C)"""

        D,L,N,M = self.round_to_possible_values(D,L,N,M)

        coord = self.get_label_coord(self.to_str_method(D,L,N,M))
        J = sum([c*dynamic_w*fixed_w for c,dynamic_w,fixed_w in zip(coord,dynamic_weights,self.weights)])
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

    def convert_res_to_lbl(self,res):
        lbl = self.to_str_method(
            *self.round_to_possible_values(
                res["params"]["D"],res["params"]["L"],res["params"]["N"],res["params"]["M"]
            )
        )
        return lbl



def bayes_lch_rate_dist(
        csv_path,axes,lambda_grid,lambdas=[],random_state=1,init_points=5,n_iter=25,ax_ranges=None,
        acquisition_func="pii"
    ):
    fixed_weights = [1 for _ in range(len(axes))]
    bayesOptRateDist = BayesOptRateDist(csv_path,axes,fixed_weights,lambdas=lambdas)

    optimizer = BayesianOptimization(
        f=bayesOptRateDist.black_box_function,
        pbounds=bayesOptRateDist.pbounds,
        verbose=2,
        random_state=random_state,
        lambda_grid=lambda_grid,
        acquisition_func=acquisition_func
    )

    optimizer.maximize(
        init_points=init_points, # default: 5
        n_iter=n_iter, # default: 25
    )

    print(optimizer.maxes)

    print({bayesOptRateDist.convert_res_to_lbl(mx) for mx in optimizer.maxes})

    cloud = bayesOptRateDist.data.loc[:,bayesOptRateDist.axes].values.tolist()

    lch_labels = [bayesOptRateDist.convert_res_to_lbl(mx) for mx in optimizer.maxes]
    visited_labels = [bayesOptRateDist.convert_res_to_lbl(res) for res in optimizer.res]
    
    visited_labels = [lbl for lbl in visited_labels if (lbl not in lch_labels)]
    cloud = [c for c,lbl in zip(cloud,bayesOptRateDist.data.index) if ((lbl not in lch_labels) and (lbl not in visited_labels))]

    lch = [bayesOptRateDist.get_label_coord(lbl) for lbl in lch_labels]
    visited = [bayesOptRateDist.get_label_coord(lbl) for lbl in visited_labels]

    from bo_utils import plot_3d_lch
    plot_3d_lch([cloud,visited,lch],["b","r","g"],['o','^','s'],[0.05,0.1,1],
        ax_ranges=ax_ranges,
        ax_labels=axes,
        planes=[{**mx, "center":bayesOptRateDist.get_label_coord(bayesOptRateDist.convert_res_to_lbl(mx))} for mx in optimizer.maxes]
    )
 


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

    lbl = bayesOptRateDist.convert_res_to_lbl(optimizer.max)

    print(
        lbl, bayesOptRateDist.get_label_coord(lbl), optimizer.max["target"]
    )

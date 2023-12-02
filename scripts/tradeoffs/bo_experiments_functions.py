"""
https://github.com/bayesian-optimization/BayesianOptimization
https://stackoverflow.com/questions/57182358/iterate-over-integers-in-bayesian-optimization-package-python
https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
"""

import random
import numpy as np
import pandas as pd
from collections.abc import Iterable

class BayesOptRateDist:

    def __init__(self,csv_path,axes,weights,lambdas=[],seed=None,normalize=True):
        self.seed=seed
        self.axes = axes
        self.weights = weights
        self.possible_values = {
            "D": [3,4],
            "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
            "N": [32, 64, 96, 128, 160, 192, 224],
            "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        }
        self.pbounds = {k:(float(v[0]),float(v[-1])) for k,v in self.possible_values.items()}
        self.normalize = normalize
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

            self.original_random_state = random.getstate()
            random.seed(self.seed)

            bpp_loss = [(
                    ((N/maxN - random.choice(self.possible_values["N"])/maxN)**2)/random.uniform(0.1, 3.9)
                    + ((M/maxM - random.choice(self.possible_values["M"])/maxM)**2)/random.uniform(0.1, 3.9)
                    + ((D/maxD - random.choice(self.possible_values["D"])/maxD)**2)/random.uniform(0.1, 3.9)
                    + max(min(np.log10(float(L)),8),-8) + 8.5
                )
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]
            mse_loss = [(
                    ((N/maxN - random.choice(self.possible_values["N"])/maxN)**2)/random.uniform(0.1, 3.9)
                    + ((M/maxM - random.choice(self.possible_values["M"])/maxM)**2)/random.uniform(0.1, 3.9)
                    + ((D/maxD - random.choice(self.possible_values["D"])/maxD)**2)/random.uniform(0.1, 3.9)
                    - max(min(np.log10(float(L)),8),-8) + 8.5
                )
                for D in self.possible_values["D"]
                for L in self.possible_values["L"]
                for N in self.possible_values["N"]
                for M in self.possible_values["M"]
            ]

            random.setstate(self.original_random_state)

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
            # data["params"] = data["params"]/1e+6

        if self.normalize:
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
        if dynamic_weights is not None:
            J = sum([c*dynamic_w*fixed_w for c,dynamic_w,fixed_w in zip(coord,dynamic_weights,self.weights)])
        else:
            J = sum([c*fixed_w for c,fixed_w in zip(coord,self.weights)])
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


def run_bo_lch(
    black_box_function,
    pbounds,
    random_state,
    lambda_grid,
    acquisition_func,
    init_points,
    n_iter
):

    from bo import BOCustom as BayesianOptimization

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=random_state,
        lambda_grid=lambda_grid,
        acquisition_func=acquisition_func
    )

    optimizer.maximize(
        init_points=init_points, # default: 5
        n_iter=n_iter, # default: 25
    )

    return optimizer.maxes, optimizer.res



def run_bo_repeatedly(
    black_box_function,
    pbounds,
    random_state,
    lambda_grid,
    acquisition_func,
    init_points,
    n_iter
):

    from bo import BOCustom as BayesianOptimization

    optimizer_maxes, optimizer_res = [],[]
    for i in range(lambda_grid.shape[0]):
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=random_state,
            lambda_grid=lambda_grid[i:i+1,:],
            acquisition_func=acquisition_func
        )

        optimizer.maximize(
            init_points=init_points, # default: 5
            n_iter=n_iter, # default: 25
        )
        optimizer_maxes.extend( optimizer.maxes )
        optimizer_res.extend( optimizer.res )
        
    return optimizer_maxes, optimizer_res



def bayes_lch_rate_dist(
        csv_path,axes,lambda_grid,lambdas=[],random_state=None,init_points=5,n_iter=25,ax_ranges=None,
        acquisition_func="pii", lch_method="jointly"
    ):
    fixed_weights = [1 for _ in range(len(axes))]
    bayesOptRateDist = BayesOptRateDist(csv_path,axes,fixed_weights,lambdas=lambdas,seed=random_state)

    if lch_method == "jointly":
        optimizer_maxes, optimizer_res = run_bo_lch(
            bayesOptRateDist.black_box_function,
            bayesOptRateDist.pbounds,
            random_state=random_state,
            lambda_grid=lambda_grid,
            acquisition_func=acquisition_func,
            init_points=init_points,
            n_iter=n_iter
        )
    elif lch_method == "repeatedly":
        optimizer_maxes, optimizer_res = run_bo_repeatedly(
            bayesOptRateDist.black_box_function,
            bayesOptRateDist.pbounds,
            random_state=random_state,
            lambda_grid=lambda_grid,
            acquisition_func=acquisition_func,
            init_points=init_points,
            n_iter=n_iter
        )
    else:
        raise ValueError(lch_method)

    print(optimizer_maxes)

    print({bayesOptRateDist.convert_res_to_lbl(mx) for mx in optimizer_maxes})

    cloud = bayesOptRateDist.data.loc[:,bayesOptRateDist.axes].values.tolist()

    lch_labels = [bayesOptRateDist.convert_res_to_lbl(mx) for mx in optimizer_maxes]
    visited_labels = [bayesOptRateDist.convert_res_to_lbl(res) for res in optimizer_res]
    
    visited_labels = [lbl for lbl in visited_labels if (lbl not in lch_labels)]
    cloud = [c for c,lbl in zip(cloud,bayesOptRateDist.data.index) if ((lbl not in lch_labels) and (lbl not in visited_labels))]

    lch = [bayesOptRateDist.get_label_coord(lbl) for lbl in set(lch_labels)]
    visited = [bayesOptRateDist.get_label_coord(lbl) for lbl in set(visited_labels)]

    from bo_utils import plot_3d_lch
    plot_3d_lch([cloud,visited,lch],["b","r","g"],['o','^','s'],[0.05,0.5,1],
        ax_ranges=ax_ranges,
        ax_labels=axes,
        planes=[{**mx, "center":bayesOptRateDist.get_label_coord(bayesOptRateDist.convert_res_to_lbl(mx))} for mx in optimizer_maxes]
    )
 


def bayes_opt_rate_dist(csv_path,axes,weights,lambdas=[],random_state=1,init_points=5,n_iter=25):

    from bayes_opt import BayesianOptimization

    bayesOptRateDist = BayesOptRateDist(csv_path,axes,weights,lambdas=lambdas,normalize=False)


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
    # print(optimizer.max)

    # list of all parameters probed and their corresponding target values 
    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))

    lbl = bayesOptRateDist.convert_res_to_lbl(optimizer.max)

    # print(
    #     lbl, bayesOptRateDist.get_label_coord(lbl), optimizer.max["target"]
    # )


    visited_labels = [bayesOptRateDist.convert_res_to_lbl(res) for res in optimizer.res]

    idx = visited_labels.index(lbl)

    n_trained_networks = len(set(visited_labels[:idx+1]))

    return lbl, n_trained_networks, optimizer.max["target"]

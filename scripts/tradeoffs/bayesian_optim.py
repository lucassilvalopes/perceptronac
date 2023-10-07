"""
https://github.com/bayesian-optimization/BayesianOptimization
https://stackoverflow.com/questions/57182358/iterate-over-integers-in-bayesian-optimization-package-python
https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
"""

import pandas as pd
from bayes_opt import BayesianOptimization


class BayesOptRateDist:

    def __init__(self,csv_path,x_axis,y_axis,lambdas=[],lmbda=1):

        self.data = self.read_data(csv_path,lambdas)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.possible_values = {
            "D": [3,4],
            "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
            "N": [32, 64, 96, 128, 160, 192, 224],
            "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
        }
        self.pbounds = {k:(float(v[0]),float(v[-1])) for k,v in self.possible_values.items()}
        self.lmbda = lmbda

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
        return self.data.loc[label,[self.x_axis,self.y_axis]].values.tolist()

    def black_box_function(self,D,L,N,M):
        """receives hyperparameters and outputs -J = -(R + lambda * D + gamma * C)"""

        D = self.round_to_possible_values(D,self.possible_values["D"])
        L = self.round_to_possible_values(L,self.possible_values["L"])
        N = self.round_to_possible_values(N,self.possible_values["N"])
        M = self.round_to_possible_values(M,self.possible_values["M"])

        x,y = self.get_label_coord(self.to_str_method(D,L,N,M))
        return -(x + self.lmbda * y)

    def round_to_possible_values(self,P,P_list):
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





if __name__ == "__main__":

    bayesOptRateDist = BayesOptRateDist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "bpp_loss","mse_loss",
        lambdas=["2e-2"],
        lmbda=2e-2*(255**2)
    )


    optimizer = BayesianOptimization(
        f=bayesOptRateDist.black_box_function,
        pbounds=bayesOptRateDist.pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.set_gp_params(alpha=1e-3)
    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    print(optimizer.max)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


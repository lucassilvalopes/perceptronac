
import numpy as np
import random
from bo_experiments_functions import bayes_lch_rate_dist
from bo_utils import simple_lambda_grid_3d,lambda_grid_3d,lambda_seq


if __name__ == "__main__":

    bayes_lch_rate_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        simple_lambda_grid_3d(),
        lambdas=["2e-2"],
        random_state=1,
        init_points=2,
        n_iter=47,
        ax_ranges=None,
        acquisition_func="pii",
        lch_method="jointly"
    )

    random_state = random.randint(0,100)

    bayes_lch_rate_dist(
        "random",
        ["bpp_loss","mse_loss","params"],
        simple_lambda_grid_3d(),
        lambdas=["2e-2"],
        random_state=random_state,
        init_points=2,
        n_iter=47,
        ax_ranges=None,
        acquisition_func="random",
        lch_method="jointly"
    )

    bayes_lch_rate_dist(
        "random",
        ["bpp_loss","mse_loss","params"],
        simple_lambda_grid_3d(),
        lambdas=["2e-2"],
        random_state=random_state,
        init_points=2,
        n_iter=47,
        ax_ranges=None,
        acquisition_func="pii",
        lch_method="jointly"

    )

    bayes_lch_rate_dist(
        "random",
        ["bpp_loss","mse_loss","params"],
        simple_lambda_grid_3d(),
        lambdas=["2e-2"],
        random_state=random_state,
        init_points=2,
        n_iter=5,
        ax_ranges=None,
        acquisition_func="pi",
        lch_method="repeatedly"
    )
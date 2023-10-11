
from bo_experiments_functions import bayes_lch_rate_dist
from bo_utils import simple_lambda_grid_3d,lambda_grid_3d,lambda_seq


if __name__ == "__main__":

    bayes_lch_rate_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        simple_lambda_grid_3d(),
        # lambda_grid_3d(y_lmbd=[0.005,0.01,0.02,500],z_lmbd=[1/1e+6]),
        # ax_ranges=[[0,4],[0.0025,0.0125],[0,9*1e6]],
        init_points=2,
        n_iter=38
    )

    # bayes_lch_rate_dist(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss","flops"],
    #     simple_lambda_grid_3d()
    # )

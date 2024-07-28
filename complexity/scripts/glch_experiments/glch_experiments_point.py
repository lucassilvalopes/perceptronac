import os
from complexity.glch_experiments_functions import glch_rate_vs_dist


if __name__ == "__main__":

    results_folder = "gho_results"
    debug_folder = "gho_debug"

    if os.path.isdir(results_folder):
        import shutil
        shutil.rmtree(results_folder)
    os.mkdir(results_folder)

    if os.path.isdir(debug_folder):
        import shutil
        shutil.rmtree(debug_folder)
    os.mkdir(debug_folder)


    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss"],
        lambdas=["2e-2"],
        algo="gho",
        constrained=False,
        weights=[1,2e-2*(255**2)],
        fldr=results_folder,
        debug_folder=debug_folder,
        debug=True
    )

    for lmbda,mult in [("5e-3",1),("5e-3",100),("1e-2",1),("1e-2",100),("2e-2",1),("2e-2",100)]:

        glch_rate_vs_dist(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            ["bpp_loss","mse_loss","params"],
            weights=[1,float(lmbda)*(255**2),1/(1e6 * mult)],
            lambdas=[lmbda],
            algo="gho",
            constrained=False,
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False
        )

        glch_rate_vs_dist(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            ["bpp_loss","mse_loss","flops"],
            weights=[1,float(lmbda)*(255**2),1/(1e10 * mult)],
            lambdas=[lmbda],
            algo="gho",
            constrained=False,
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False
        )

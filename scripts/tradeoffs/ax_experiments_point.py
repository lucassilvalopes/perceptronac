
import numpy as np
from functools import partial
from ax_experiments_functions import rdc_loss_setup, rdc_read_glch_data, rdc_label_to_params
from ax_utils import ax_glch_comparison_sohpo


N_SEEDS = 25
SEEDS_RANGE = [1, 10000]
N_INIT = 6

if __name__ == "__main__":

    for lmbda,mult in [("5e-3",1),("5e-3",100),("1e-2",1),("1e-2",100),("2e-2",1),("2e-2",100)]:

        lambdas=[lmbda]
        formatted_lambdas = "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])+"_"

        weights=[1,float(lmbda)*(255**2),1/(1e6 * mult)]
        formatted_weights = f"{'_'.join(['{:.0e}'.format(w) for w in weights])}"
        
        ax_glch_comparison_sohpo(
            results_folder="ax_results",
            data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018"+\
                "-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            setup_func=partial(rdc_loss_setup,weights=weights,lambdas=lambdas,complexity_axis="params"),
            glch_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results/"+\
                f"glch1D_weights_{formatted_weights}_lambdas_{formatted_lambdas}_bpp_loss_vs_mse_loss_vs_params_start_left_history.csv",
            read_glch_data_func=partial(rdc_read_glch_data,complexity_axis="params"),
            label_to_params_func=rdc_label_to_params,
            n_seeds = N_SEEDS,
            seeds_range = SEEDS_RANGE,
            n_init=N_INIT
        )

        weights=[1,float(lmbda)*(255**2),1/(1e10 * mult)]
        formatted_weights = f"{'_'.join(['{:.0e}'.format(w) for w in weights])}"

        ax_glch_comparison_sohpo(
            results_folder="ax_results",
            data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018"+\
                "-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            setup_func=partial(rdc_loss_setup,weights=weights,lambdas=lambdas,complexity_axis="flops"),
            glch_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results/"+\
                f"glch1D_weights_{formatted_weights}_lambdas_{formatted_lambdas}_bpp_loss_vs_mse_loss_vs_flops_start_left_history.csv",
            read_glch_data_func=partial(rdc_read_glch_data,complexity_axis="flops"),
            label_to_params_func=rdc_label_to_params,
            n_seeds = N_SEEDS,
            seeds_range = SEEDS_RANGE,
            n_init=N_INIT
        )


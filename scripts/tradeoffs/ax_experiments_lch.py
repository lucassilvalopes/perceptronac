
import os
import random
import numpy as np
import pandas as pd
from ax_utils import sobol_method,ehvi_method,parego_method,get_summary_df
from ax_utils import get_init_hv_list,plot_mohpo_methods
from ax_experiments_functions import ax_rdc_setup, get_glch_hv_list_rdc



def ax_rdp(csv_path,results_folder,n_seeds):

    original_random_state = random.getstate()
    random.seed(42)
    random_seeds = random.sample(range(1, 10000), n_seeds)
    random.setstate(original_random_state)

    search_space,optimization_config,MAX_HV = ax_rdc_setup(csv_path)

    N_INIT = 6
    N_BATCH = 105

    for seed in random_seeds:

        sobol_hv_list = sobol_method(search_space,optimization_config,seed,N_INIT,N_BATCH)

        ehvi_hv_list = ehvi_method(search_space,optimization_config,seed,N_INIT,N_BATCH)

        parego_hv_list = parego_method(search_space,optimization_config,seed,N_INIT,N_BATCH)

        init_hv_list = get_init_hv_list(search_space,optimization_config,seed,N_INIT)

        iters = np.arange(1, N_INIT + N_BATCH + 1)
        methods_df = get_summary_df(iters,init_hv_list,sobol_hv_list,ehvi_hv_list,parego_hv_list,len(iters)*[None],MAX_HV)
        methods_df.to_csv(f"{results_folder}/bpp_loss_mse_loss_params_ax_methods_seed{seed}.csv")



if __name__ == "__main__":

    results_folder = "ax_results"
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    ax_rdp(
        csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        results_folder=results_folder,
        n_seeds = 25
    )





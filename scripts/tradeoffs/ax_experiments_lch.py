
import os
import random
import numpy as np
import pandas as pd
from ax_utils import sobol_method,ehvi_method,parego_method,get_summary_df
from ax_utils import get_init_hv_list,plot_mohpo_methods, combine_results
from ax_experiments_functions import ax_rdc_setup, get_glch_hv_list_rdc



def ax_rdc(data_csv_path,complexity_axis,glch_csv_path,results_folder,n_seeds,seeds_range = [1, 10000],n_init=6):

    original_random_state = random.getstate()
    random.seed(42)
    random_seeds = random.sample(range(*seeds_range), n_seeds)
    random.setstate(original_random_state)

    search_space,optimization_config,max_hv = ax_rdc_setup(data_csv_path)

    glch_hv_list = get_glch_hv_list_rdc(search_space,optimization_config,glch_csv_path,complexity_axis)

    n_batch = len(glch_hv_list) - n_init

    for seed in random_seeds:

        sobol_hv_list = sobol_method(search_space,optimization_config,seed,n_init,n_batch)

        ehvi_hv_list = ehvi_method(search_space,optimization_config,seed,n_init,n_batch)

        parego_hv_list = parego_method(search_space,optimization_config,seed,n_init,n_batch)

        init_hv_list = get_init_hv_list(search_space,optimization_config,seed,n_init)

        iters = np.arange(1, n_init + n_batch + 1)
        methods_df = get_summary_df(iters,init_hv_list,sobol_hv_list,ehvi_hv_list,parego_hv_list,len(iters)*[None],max_hv)
        methods_df.to_csv(f"{results_folder}/bpp_loss_mse_loss_{complexity_axis}_ax_methods_seed{seed}.csv")


    avg_df = combine_results(results_folder,glch_hv_list)

    plot_mohpo_methods(avg_df,f"{results_folder}/bpp_loss_mse_loss_{complexity_axis}_ax_methods_avgs.png")




if __name__ == "__main__":

    results_folder = "ax_results"
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    ax_rdc(
        data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        complexity_axis = "params",
        glch_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/threed_hull_bpp_loss_vs_mse_loss_vs_params_start_left.csv",
        results_folder=results_folder,
        n_seeds = 1,
        seeds_range = [1234,1235]
    )





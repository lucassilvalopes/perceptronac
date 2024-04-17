
from functools import partial
from ax_experiments_functions import rdc_loss_setup, rdc_read_glch_data, rdc_label_to_params
from ax_utils import ax_glch_comparison_sohpo


N_SEEDS = 25
SEEDS_RANGE = [1, 10000]
N_INIT = 6

if __name__ == "__main__":

    ax_glch_comparison_sohpo(
        results_folder="ax_results_rdc_params",
        data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018"+\
            "-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        setup_func=partial(rdc_loss_setup,weights=[1,2e-2*(255**2),1/(1e6 * 100)],lambdas=["2e-2"],complexity_axis="params"),
        glch_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results/"+\
            "glch1D_weights_1e+00_1e+03_1e-08_lambdas_2e-2_bpp_loss_vs_mse_loss_vs_params_start_left_history.csv",
        read_glch_data_func=rdc_read_glch_data,
        label_to_params_func=rdc_label_to_params,
        n_seeds = N_SEEDS,
        seeds_range = SEEDS_RANGE,
        n_init=N_INIT
    )

import os
from ax_utils import ax_glch_comparison
from functools import partial
from ax_experiments_functions import rdc_setup,rdc_read_glch_data,rdc_label_to_params


if __name__ == "__main__":

    ax_glch_comparison(
        results_folder="ax_results_rdc_params",
        data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        setup_func=partial(rdc_setup,complexity_axis="params"),
        glch_csv_paths = {
            "c_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "u_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_unconstrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "c_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "u_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_unconstrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "c_tie_break": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_tie_break_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv"
        },
        read_glch_data_func=rdc_read_glch_data,
        label_to_params_func=rdc_label_to_params,
        n_seeds = 1,
        seeds_range = [1234,1235],
        n_init=6
    )








import os
from ax_experiments_functions import ax_rdc


if __name__ == "__main__":

    results_folder = "ax_results"
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    ax_rdc(
        data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        complexity_axis = "params",
        glch_csv_paths = {
            "c_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "u_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_unconstrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "c_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "u_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_unconstrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv",
            "c_tie_break": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_tie_break_constrained_bpp_loss_vs_mse_loss_vs_params_start_left_threed_history.csv"
        }
        results_folder=results_folder,
        n_seeds = 1,
        seeds_range = [1234,1235]
    )





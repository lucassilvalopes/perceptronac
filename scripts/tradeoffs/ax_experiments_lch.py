
import os
from ax_utils import ax_glch_comparison_mohpo
from functools import partial
from ax_experiments_functions import rdc_setup,rdc_read_glch_data,rdc_label_to_params
from ax_experiments_functions import rc_setup,rc_read_glch_data,rc_label_to_params
from ax_experiments_functions import rb_setup,rb_read_glch_data,rb_label_to_params


N_SEEDS = 25
SEEDS_RANGE = [1, 10000]
N_INIT = 6


if __name__ == "__main__":

    for complexity_axis in ["params","flops"]:

        ax_glch_comparison_mohpo(
            results_folder="ax_results",
            data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            setup_func=partial(rdc_setup,complexity_axis=complexity_axis),
            glch_csv_paths = {
                "c_angle_rule": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "u_angle_rule": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_angle_rule_unconstrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "c_gift_wrapping": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "u_gift_wrapping": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_gift_wrapping_unconstrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "c_tie_break": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch3D_tie_break_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv"
            },
            read_glch_data_func=rdc_read_glch_data,
            label_to_params_func=rdc_label_to_params,
            n_seeds = N_SEEDS,
            seeds_range = SEEDS_RANGE,
            n_init=N_INIT
        )

    for complexity_axis in ["energy_noisy","params"]:

        ax_glch_comparison_mohpo(
            results_folder="ax_results",
            data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv",
            setup_func=partial(rc_setup,complexity_axis=("micro_joules_per_pixel" if complexity_axis == "energy_noisy" else complexity_axis)),
            glch_csv_paths = {
                "c_angle_rule": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_angle_rule_constrained_rate_vs_{complexity_axis}_history.csv",
                "u_angle_rule": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_angle_rule_unconstrained_rate_vs_{complexity_axis}_history.csv",
                "c_gift_wrapping": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_gift_wrapping_constrained_rate_vs_{complexity_axis}_history.csv",
                "u_gift_wrapping": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_gift_wrapping_unconstrained_rate_vs_{complexity_axis}_history.csv",
                "c_tie_break": f"/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_tie_break_constrained_rate_vs_{complexity_axis}_history.csv"
            },
            read_glch_data_func=rc_read_glch_data,
            label_to_params_func=rc_label_to_params,
            n_seeds = N_SEEDS,
            seeds_range = SEEDS_RANGE,
            n_init=N_INIT
        )

    ax_glch_comparison_mohpo(
        results_folder="ax_results",
        data_csv_path = "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-model-bits_hx-10-20-40-80-160-320-640_b-8-16-32.csv",
        setup_func=rb_setup,
        glch_csv_paths = {
            "c_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_angle_rule_constrained_model_bits_vs_data_bits_history.csv",
            "u_angle_rule": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_angle_rule_unconstrained_model_bits_vs_data_bits_history.csv",
            "c_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_gift_wrapping_constrained_model_bits_vs_data_bits_history.csv",
            "u_gift_wrapping": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_gift_wrapping_unconstrained_model_bits_vs_data_bits_history.csv",
            "c_tie_break": "/home/lucas/Documents/perceptronac/scripts/tradeoffs/glch_results/glch2D_tie_break_constrained_model_bits_vs_data_bits_history.csv"
        },
        read_glch_data_func=rb_read_glch_data,
        label_to_params_func=rb_label_to_params,
        n_seeds = N_SEEDS,
        seeds_range = SEEDS_RANGE,
        n_init=N_INIT
    )






from functools import partial
from complexity.hypervolume import adjust_ax_glch_comparison_mohpo
from complexity.ax_experiments_functions import rc_setup, rc_read_glch_data, rc_label_to_params, rc_params_to_label
from complexity.ax_experiments_functions import rb_setup,rb_read_glch_data,rb_label_to_params, rb_params_to_label
from complexity.ax_experiments_functions import rdc_setup,rdc_read_glch_data,rdc_label_to_params, rdc_params_to_label
from complexity.hypervolume_support import lim_ax_methods_dfs_with_counts_and_save
from complexity.hypervolume_support import lim_ax_methods_dfs_with_counts_and_save_rdc
from complexity.hypervolume_graphics import gen_all_graphs, gen_all_graphs_rdc



for up_to_complexity in [False,True]:
    for complexity_axis in ["energy_noisy","params"]:

        all_hvs_df = adjust_ax_glch_comparison_mohpo(
            ax_results_folder = "/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_energy_params_bits/",
            data_csv_path = "/home/lucas/Documents/perceptronac/complexity/data/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv",
            setup_func = partial(rc_setup,complexity_axis=("micro_joules_per_pixel" if complexity_axis == "energy_noisy" else complexity_axis)),
            glch_csv_paths = {
                "c_angle_rule": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_angle_rule_constrained_rate_vs_{complexity_axis}_history.csv",
                "u_angle_rule": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_angle_rule_unconstrained_rate_vs_{complexity_axis}_history.csv",
                "c_gift_wrapping": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_gift_wrapping_constrained_rate_vs_{complexity_axis}_history.csv",
                "u_gift_wrapping": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_gift_wrapping_unconstrained_rate_vs_{complexity_axis}_history.csv",
                "c_tie_break": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_tie_break_constrained_rate_vs_{complexity_axis}_history.csv"
            },
            read_glch_data_func = rc_read_glch_data,
            label_to_params_func = rc_label_to_params,
            params_to_label_func = rc_params_to_label,
            up_to_complexity=up_to_complexity
        )







for up_to_complexity in [False,True]:
    all_hvs_df = adjust_ax_glch_comparison_mohpo(
        ax_results_folder="/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_energy_params_bits/",
        data_csv_path = "/home/lucas/Documents/perceptronac/complexity/data/rate-model-bits_hx-10-20-40-80-160-320-640_b-8-16-32.csv",
        setup_func=rb_setup,
        glch_csv_paths = {
            "c_angle_rule": "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_angle_rule_constrained_model_bits_vs_data_bits_history.csv",
            "u_angle_rule": "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_angle_rule_unconstrained_model_bits_vs_data_bits_history.csv",
            "c_gift_wrapping": "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_gift_wrapping_constrained_model_bits_vs_data_bits_history.csv",
            "u_gift_wrapping": "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_gift_wrapping_unconstrained_model_bits_vs_data_bits_history.csv",
            "c_tie_break": "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch2D_tie_break_constrained_model_bits_vs_data_bits_history.csv"
        },
        read_glch_data_func=rb_read_glch_data,
        label_to_params_func=rb_label_to_params,
        params_to_label_func = rb_params_to_label,
        up_to_complexity=up_to_complexity
    )







for up_to_complexity in [False,True]:
    for complexity_axis in ["params","flops"]:

        all_hvs_df = adjust_ax_glch_comparison_mohpo(
            ax_results_folder=f"/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_rdc_{complexity_axis}",
            data_csv_path = "/home/lucas/Documents/perceptronac/complexity/data/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            setup_func=partial(rdc_setup,complexity_axis=complexity_axis),
            glch_csv_paths = {
                "c_angle_rule": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch3D_angle_rule_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "u_angle_rule": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch3D_angle_rule_unconstrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "c_gift_wrapping": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch3D_gift_wrapping_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "u_gift_wrapping": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch3D_gift_wrapping_unconstrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv",
                "c_tie_break": f"/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/glch_results/glch3D_tie_break_constrained_bpp_loss_vs_mse_loss_vs_{complexity_axis}_start_left_threed_history.csv"
            },
            read_glch_data_func=partial(rdc_read_glch_data,complexity_axis=complexity_axis),
            label_to_params_func=rdc_label_to_params,
            params_to_label_func = rdc_params_to_label,
            up_to_complexity=up_to_complexity
        )







for up_to_complexity in [False,True]:
    for min_support in [10]:
        for x_axis in ["micro_joules_per_pixel","model_bits","params"]:
            lim_ax_methods_dfs_with_counts_and_save(x_axis,
                "/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_energy_params_bits/",
                "/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/",
                min_support,up_to_complexity)





for up_to_complexity in [False]:
    for min_support in [10]:
        for x_axis in ["params","flops"]:
            lim_ax_methods_dfs_with_counts_and_save_rdc(x_axis,
                f"/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_rdc_{x_axis}",
                "/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/",
                min_support,up_to_complexity)



gen_all_graphs("/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/")


gen_all_graphs_rdc("/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/")



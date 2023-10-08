import os
from glch_functions import (
    glch_rate_vs_energy,glch_rate_vs_params,glch_rate_vs_time,
    glch_rate_vs_dist,glch_rate_vs_dist_2,glch_model_bits_vs_data_bits
)


if __name__ == "__main__":

    results_folder = "glch_results"
    debug_folder = "glch_debug"

    if os.path.isdir(results_folder):
        import shutil
        shutil.rmtree(results_folder)
    os.mkdir(results_folder)

    if os.path.isdir(debug_folder):
        import shutil
        shutil.rmtree(debug_folder)
    os.mkdir(debug_folder)

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "micro_joules_per_pixel",
        "data_bits/data_samples",
        "rate_vs_energy",
        # x_range=[135,175],
        # y_range=[0.115,0.145],
        x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
        y_alias="Rate (bits per pixel)",
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "micro_joules_per_pixel",
        "data_bits/data_samples",
        "rate_vs_energy_noisy",
        # x_range=[140,180],
        # y_range=None,
        remove_noise=False,
        x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
        y_alias="Rate (bits per pixel)",
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_params(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "params","data_bits/data_samples",
        "rate_vs_params",
        # x_range=None,
        # y_range=None,
        x_in_log_scale=True,
        x_alias="Complexity (multiply/adds per pixel)",
        y_alias="Rate (bits per pixel)",
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    # TODO: the time measurements right now are too comprehensive.
    # They are measuring more than just the network computations.
    # They are also measuring the time taken to load the data, etc.
    # I could try to restrict the time measurements a bit more.
    # In other words, the measurements seem a little biased.
    glch_rate_vs_time(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "time","data_bits/data_samples",
        "rate_vs_time",
        # x_range=None,
        # y_range=None,
        remove_noise=False,
        x_in_log_scale=False,
        x_alias="Complexity (seconds)",
        y_alias="Rate (bits per pixel)",
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss"],
        # axes_ranges=[[0.1,1.75],[0.001,0.0045]],
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_dist_2(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "bpp_loss","mse_loss",#1,1,
        # x_range=[0.1,1.75],
        # y_range=[0.001,0.0045],
        start="left",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["flops","loss"],
        # axes_ranges=[[-0.2*1e10,3.75*1e10],[1.1,3.1]],
        lambdas=["2e-2"],
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["params","loss"],
        # axes_ranges=[[-0.1*1e6,4*1e6],[1.1,3.1]],
        lambdas=["2e-2"],
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

    glch_model_bits_vs_data_bits(
        "/home/lucas/Documents/perceptronac/results/exp_1676160183/exp_1676160183_model_bits_x_data_bits_values.csv",
        "model_bits","data_bits/data_samples",
        # x_range=[-0.1,0.8],
        # y_range=None,
        x_in_log_scale=True,
        x_alias="Complexity (encoded model bits)",
        y_alias="Rate (bits per pixel)",
        algo="glch",
        constrained=True,
        fldr=results_folder,
        debug_folder=debug_folder
    )

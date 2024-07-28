import os
from complexity.glch_experiments_functions import (
    glch_rate_vs_energy,glch_rate_vs_params,glch_rate_vs_time,
    glch_rate_vs_dist,glch_model_bits_vs_data_bits,
    glch3d_rdc
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

    for select_function,constrained in [
        ("tie_break",True),
        ("gift_wrapping",True),
        ("gift_wrapping",False),
        ("angle_rule",True),
        ("angle_rule",False)
    ]:

        # glch_rate_vs_energy(
        #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-joules-time-params_hx-10-20-40-80-160-320-640.csv",
        #     "micro_joules_per_pixel",
        #     "data_bits/data_samples",
        #     "rate_vs_energy",
        #     x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
        #     y_alias="Rate (bits per pixel)",
        #     algo="glch",
        #     constrained=constrained,
        #     fldr=results_folder,
        #     debug_folder=debug_folder,
        #     debug=False,
        #     select_function=select_function
        # )

        glch_rate_vs_energy(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv",
            "micro_joules_per_pixel",
            "data_bits/data_samples",
            "rate_vs_energy_noisy",
            x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
            y_alias="Rate (bits per pixel)",
            algo="glch",
            constrained=constrained,
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False,
            select_function=select_function
        )

        glch_rate_vs_params(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv",
            "params","data_bits/data_samples",
            "rate_vs_params",
            x_in_log_scale=True,
            x_alias="Complexity (multiply/adds per pixel)",
            y_alias="Rate (bits per pixel)",
            algo="glch",
            constrained=constrained,
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False,
            select_function=select_function
        )

        # # TODO: the time measurements right now are too comprehensive.
        # # They are measuring more than just the network computations.
        # # They are also measuring the time taken to load the data, etc.
        # # I could try to restrict the time measurements a bit more.
        # # In other words, the measurements seem a little biased.
        # glch_rate_vs_time(
        #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv",
        #     "time","data_bits/data_samples",
        #     "rate_vs_time",
        #     x_in_log_scale=False,
        #     x_alias="Complexity (seconds)",
        #     y_alias="Rate (bits per pixel)",
        #     algo="glch",
        #     constrained=constrained,
        #     fldr=results_folder,
        #     debug_folder=debug_folder,
        #     debug=False,
        #     select_function=select_function
        # )

        # glch_rate_vs_dist(
        #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        #     ["bpp_loss","mse_loss"],
        #     algo="glch",
        #     constrained=constrained,
        #     fldr=results_folder,
        #     debug_folder=debug_folder,
        #     debug=False,
        #     select_function=select_function
        # )



        glch3d_rdc(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            complexity_axis="flops",
            constrained=constrained,
            start="left",
            lambdas=["5e-3", "1e-2", "2e-2"],
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False,
            select_function=select_function,
            axes_aliases=["Complexity (FLOPs)","$L=R+\lambda D$"]
        )

        glch3d_rdc(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            complexity_axis="params",
            constrained=constrained,
            start="left",
            lambdas=["5e-3", "1e-2", "2e-2"],
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False,
            select_function=select_function,
            axes_aliases=["Complexity (number of parameters)","$L=R+\lambda D$"]
        )

        glch_model_bits_vs_data_bits(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-model-bits_hx-10-20-40-80-160-320-640_b-8-16-32.csv",
            "model_bits","data_bits/data_samples",
            x_in_log_scale=True,
            x_alias="Complexity (encoded model bits)",
            y_alias="Rate (bits per pixel)",
            algo="glch",
            constrained=constrained,
            fldr=results_folder,
            debug_folder=debug_folder,
            debug=False,
            select_function=select_function
        )

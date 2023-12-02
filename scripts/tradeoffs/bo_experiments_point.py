import random
from bo_experiments_functions import bayes_opt_rate_dist

if __name__ == "__main__":

    original_random_state = random.getstate()
    random.seed(42)
    random_states = random.sample(range(1, 10000), 10)
    random.setstate(original_random_state)

    print(random_states)

    results_list = []
    for random_state in random_states:

        lbl,n_trained_networks,idx = bayes_opt_rate_dist(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            ["bpp_loss","mse_loss"],
            [1,2e-2*(255**2)],
            lambdas=["2e-2"],
            random_state=random_state
        )
        results_list.append((lbl,n_trained_networks,idx))
    print(results_list)

    # bayes_opt_rate_dist(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss","params"],
    #     [1,2e-2*(255**2),1/1e6],
    #     lambdas=["2e-2"]
    # )

    # bayes_opt_rate_dist(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss","flops"],
    #     [1,2e-2*(255**2),1/1e10],
    #     lambdas=["2e-2"]
    # )

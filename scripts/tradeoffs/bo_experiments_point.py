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

        lbl,n_trained_networks,target = bayes_opt_rate_dist(
            "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
            ["bpp_loss","mse_loss"],
            [1,2e-2*(255**2)],
            lambdas=["2e-2"],
            random_state=random_state
        )
        results_list.append((lbl,n_trained_networks,target))
    
    n_hits = sum([(1 if r[0] == "D3L2e-2N160M32" else 0) for r in results_list])
    n_misses = sum([(1 if r[0] != "D3L2e-2N160M32" else 0) for r in results_list])
    avg_n_trained_networks = sum([r[1] for r in results_list])/len(results_list)

    best_target = -results_list[([r[0] for r in results_list]).index("D3L2e-2N160M32")][2]

    percent_higher = sum([(-r[2] - best_target)/best_target for r in results_list])/len(results_list)

    print(f"number of hits: {n_hits}")
    print(f"average number of trained networks: {avg_n_trained_networks}")
    print(f"best target: {best_target}")
    print(f"target higher on average by (%): {percent_higher}")

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

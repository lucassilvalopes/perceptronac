
import os
import numpy as np
import random
from bo_experiments_functions import bayes_opt_rate_dist

RESULTS_FOLDER = "bo_results"


def bo_statistics(*args,**kwargs):

    original_random_state = random.getstate()
    random.seed(42)
    random_states = random.sample(range(1, 10000), 10)
    random.setstate(original_random_state)

    print(random_states)

    results_list = []
    for random_state in random_states:

        kwargs["random_state"] = random_state

        lbl,target,n_trained_networks,optimal_point_lbl,optimal_point_target = \
            bayes_opt_rate_dist(*args,**kwargs)
        results_list.append((
            lbl,n_trained_networks,target,optimal_point_lbl,optimal_point_target))
    
    n_hits = sum([(1 if r[0] == optimal_point_lbl else 0) for r in results_list])
    avg_n_trained_networks = sum([r[1] for r in results_list])/len(results_list)

    percent_higher = sum([(-r[2] - optimal_point_target)/optimal_point_target for r in results_list])/len(results_list)

    axes = args[1]
    lambdas = kwargs["lambdas"]
    formatted_lambdas = "" if len(lambdas)==0 else "_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])

    exp_id = f'{"_vs_".join(axes)}{formatted_lambdas}'

    with open(f'{RESULTS_FOLDER}/tree_{exp_id}.txt', 'w') as f:

        print(f"number of hits: {n_hits} out of {len(results_list)} trials",file=f)
        print(f"average number of trained networks: {avg_n_trained_networks}",file=f)
        print(f"best target: {optimal_point_target}",file=f)
        print(f"target higher on average by (%): {percent_higher}",file=f)


if __name__ == "__main__":

    if os.path.isdir(RESULTS_FOLDER):
        import shutil
        shutil.rmtree(RESULTS_FOLDER)
    os.mkdir(RESULTS_FOLDER)

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss"],
        [1,2e-2*(255**2)],
        lambdas=["2e-2"]
    )

    # bo_statistics(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss","params"],
    #     [1,2e-2*(255**2),1/1e6],
    #     lambdas=["2e-2"]
    # )

    # bo_statistics(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss","flops"],
    #     [1,2e-2*(255**2),1/1e10],
    #     lambdas=["2e-2"]
    # )

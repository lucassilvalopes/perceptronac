
import os
import numpy as np
import pandas as pd
import random
from bo_experiments_functions import bayes_opt_rate_dist

RESULTS_FOLDER = "bo_results"


def bo_statistics(*args,**kwargs):

    original_random_state = random.getstate()
    random.seed(42)
    random_states = random.sample(range(1, 10000), 10)
    random_states = random_states[:4]
    random.setstate(original_random_state)

    print(random_states)

    results_list = []
    min_loss_histories = []
    for random_state in random_states:

        kwargs["random_state"] = random_state

        kwargs["n_iter"] = 500

        lbl,loss,n_trained_networks,optimal_point_lbl,optimal_point_loss,min_loss_history,n_trained_networks_history = \
            bayes_opt_rate_dist(*args,**kwargs)
        results_list.append((
            lbl,loss,n_trained_networks,optimal_point_lbl,optimal_point_loss))
        
        assert len(min_loss_history) >= 23, len(min_loss_history)

        min_loss_histories.append(min_loss_history[:23])
    
    n_hits = sum([(1 if r[0] == optimal_point_lbl else 0) for r in results_list])
    avg_n_trained_networks = sum([r[2] for r in results_list])/len(results_list)

    avg_loss = sum([r[1] for r in results_list])/len(results_list) 

    percent_higher = 100 * sum([(r[1] - optimal_point_loss)/optimal_point_loss for r in results_list])/len(results_list)

    axes = args[1]
    lambdas = kwargs["lambdas"]
    formatted_lambdas = "" if len(lambdas)==0 else "_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])

    exp_id = f'{"_vs_".join(axes)}{formatted_lambdas}'

    with open(f'{RESULTS_FOLDER}/optimal_point_{exp_id}.txt', 'w') as f:

        print(f"number of hits: {n_hits} out of {len(results_list)} trials",file=f)
        print(f"average number of trained networks: {avg_n_trained_networks}",file=f)
        print(f"average estimated best loss: {avg_loss}",file=f)
        print(f"true best loss: {optimal_point_loss}",file=f)
        print(f"average loss higher by (%): {percent_higher}",file=f)

    csv_data = np.array(min_loss_histories).T
    df = pd.DataFrame(data=csv_data,index=list(range(1,csv_data.shape[0]+1)),columns=random_states)
    df["mean"] = df.mean(axis=1)
    df.to_csv(f'{RESULTS_FOLDER}/optimal_point_{exp_id}.csv')


if __name__ == "__main__":

    if os.path.isdir(RESULTS_FOLDER):
        import shutil
        shutil.rmtree(RESULTS_FOLDER)
    os.mkdir(RESULTS_FOLDER)

    # bo_statistics(
    #     "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
    #     ["bpp_loss","mse_loss"],
    #     [1,2e-2*(255**2)],
    #     lambdas=["2e-2"]
    # )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        [1,5e-3*(255**2),1/(1e6 * 100)],
        lambdas=["5e-3"]
    )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","flops"],
        [1,5e-3*(255**2),1/(1e10 * 100)],
        lambdas=["5e-3"]
    )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        [1,1e-2*(255**2),1/(1e6 * 100)],
        lambdas=["1e-2"]
    )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","flops"],
        [1,1e-2*(255**2),1/(1e10 * 100)],
        lambdas=["1e-2"]
    )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","params"],
        [1,2e-2*(255**2),1/(1e6 * 100)],
        lambdas=["2e-2"]
    )

    bo_statistics(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_D-3-4_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        ["bpp_loss","mse_loss","flops"],
        [1,2e-2*(255**2),1/(1e10 * 100)],
        lambdas=["2e-2"]
    )


"""
python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_start_left_5e-3.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_5e-3.csv

python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_start_left_1e-2.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_1e-2.csv

python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_start_left_2e-2.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_2e-2.csv

python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_start_left_5e-3.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_5e-3.csv

python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_start_left_1e-2.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_1e-2.csv

python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_start_left_2e-2.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_flops_2e-2.csv
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    glch_csv= sys.argv[1]
    bo_csv = sys.argv[2]

    df_glch = pd.read_csv(glch_csv,index_col=0)
    df_bo = pd.read_csv(bo_csv,index_col=0)

    with open(glch_csv.replace(".csv",".txt"),"r") as f:
        s = f.read()
    true_best = float(re.search(r'(?<=true best loss:).*(?=\n)',s).group())

    glch_x = df_glch["n_trials"]
    glch_y = df_glch["loss"]

    bo_x = df_bo.index
    bo_y = df_bo["mean"]

    true_best_x = range(1,max(np.max(glch_x),np.max(bo_x))+1)
    true_best_y = [true_best] * len(true_best_x)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    h1, = ax.plot(glch_x,glch_y,linestyle="",color="black",label="GLCH",marker="o")
    h2, = ax.plot(bo_x,bo_y,linestyle="",color="red",label="BO",marker="x")
    h3, = ax.plot(true_best_x,true_best_y,linestyle="dashed",color="black",label="best",marker="",linewidth=1)
    ax.legend(handles=[h1,h2,h3],loc="upper right")
    ax.set_xlabel("trials", fontsize=16)
    ax.set_ylabel("$R + \lambda_0 D + \gamma_0 C$", fontsize=16)
    fig.savefig(
        glch_csv.replace(".csv",".png"), 
        dpi=300, facecolor='w', bbox_inches = "tight")
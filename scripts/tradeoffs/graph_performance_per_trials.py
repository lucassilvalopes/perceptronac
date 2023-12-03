
"""
python graph_performance_per_trials.py /home/lucas/Documents/perceptronac/scripts/tradeoffs/gho_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_start_left_5e-3.csv /home/lucas/Documents/perceptronac/scripts/tradeoffs/bo_results_gamma_1e-12_for_flops_1e-8_for_params/optimal_point_bpp_loss_vs_mse_loss_vs_params_5e-3.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    glch_csv= sys.argv[1]
    bo_csv = sys.argv[2]

    df_glch = pd.read_csv(glch_csv,index_col=0)
    df_bo = pd.read_csv(bo_csv,index_col=0)

    glch_x = df_glch["n_trials"]
    glch_y = df_glch["loss"]

    bo_x = df_bo.index
    bo_y = df_bo["mean"]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    h1, = ax.plot(glch_x,glch_y,linestyle="",color="black",label="GLCH",marker="o")
    h2, = ax.plot(bo_x,bo_y,linestyle="",color="red",label="BO",marker="x")
    ax.legend(handles=[h1,h2],loc="upper right")
    ax.set_xlabel("trials", fontsize=16)
    ax.set_ylabel("$R + \lambda_0 D + \gamma_0 C$", fontsize=16)
    fig.savefig(
        glch_csv.replace(".csv",".png"), 
        dpi=300, facecolor='w', bbox_inches = "tight")
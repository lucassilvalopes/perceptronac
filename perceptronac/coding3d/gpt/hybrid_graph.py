
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # template = "results/ricardo10_frame0000_GPT_Q{}_blocksize8_rho95e-2_gptgpcc_results.csv"
    template = "results/longdress_vox10_1300_GPT_Q{}_blocksize8_rho95e-2_gptgpcc_results.csv"

    filename = os.path.splitext(os.path.basename(template))[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    hybrid_rates = []
    hybrid_dists = []
    gpt_rates = []
    gpt_dists = []
    for i,Q in enumerate([10,20,30,40]):

        df = pd.read_csv(template.format(Q))

        if i == 0:

            h1,= ax.plot(df["gpcc_rate"],df["gpcc_dist"], linestyle="solid",label="gpcc",color="r",marker="o")
        
        hybrid_rates.append(df["hybrid_rate"].values[-1])
        hybrid_dists.append(df["hybrid_dist"].values[-1])
        gpt_rates.append(df["gpt_rate"].values[-1])
        gpt_dists.append(df["gpt_dist"].values[-1])

    h2,= ax.plot(hybrid_rates,hybrid_dists,
        linestyle="dashed",label="hybrid",color="b",marker="^")
    h3,= ax.plot(gpt_rates,gpt_dists,
        linestyle="dotted",label="gpt",color="g",marker="s")
    ax.legend(handles=[h1,h2,h3],loc="upper right")
    ax.set_xlabel("bpov yuv")
    ax.set_ylabel("psnr y")
    fig.savefig(f"results/{filename.format('10-20-30-40')}.png", dpi=300, facecolor='w')

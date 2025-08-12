
"""
import pandas as pd

df = pd.read_csv("bpp_loss_mse_loss_flops_ax_methods_avgs_adjusted_support10.csv")
df2 = df.drop(['c_angle_rule', 'u_angle_rule','c_tie_break'],axis=1).set_index("iters")
df2.to_csv("paper_bpp_loss_mse_loss_flops_ax_methods_avgs_adjusted_support10.csv")

df = pd.read_csv("bpp_loss_mse_loss_params_ax_methods_avgs_adjusted_support10.csv")
df2 = df.drop(['c_angle_rule', 'u_angle_rule','c_tie_break'],axis=1).set_index("iters")
df2.to_csv("paper_bpp_loss_mse_loss_params_ax_methods_avgs_adjusted_support10.csv")
"""


from complexity.hypervolume_graphics import mohpo_grph_impl
import matplotlib.pyplot as plt


fldr = "/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/"

y_axis_units="Log Hypervolume Difference"

col_lbls_map = {
    "sobol": "Sobol", 
    "ehvi": "qNEHVI",
    "parego": "qNParEGO",
    "c_gift_wrapping": "GLCH-CS",
    "u_gift_wrapping": "GLCH-US"}

fig, axes= plt.subplots(nrows=2, ncols=1, figsize=(6,10.2))
for i,x_axis,title in zip([0,1],["params","flops"],["(a)","(b)"]):
    fil3 = f"paper_bpp_loss_mse_loss_{x_axis}_ax_methods_avgs_adjusted_support10.csv"
    mohpo_grph_impl(axes[i],fldr+fil3,None,False,y_axis_units,col_lbls_map)
    axes[i].set_title(title, y=-0.275, fontsize=18)

fig.subplots_adjust(hspace=0.3)
fig.savefig("glch_vs_ax.png", dpi=300, facecolor='w', bbox_inches = "tight")




        

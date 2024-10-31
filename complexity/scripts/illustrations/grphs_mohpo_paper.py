
"""
import pandas as pd

df = pd.read_csv("bpp_loss_mse_loss_flops_ax_methods_avgs_adjusted_support10.csv")
df2 = df.drop(['c_angle_rule', 'u_angle_rule','c_tie_break'],axis=1).set_index("iters")
df2.to_csv("paper_bpp_loss_mse_loss_flops_ax_methods_avgs_adjusted_support10.csv")

df = pd.read_csv("bpp_loss_mse_loss_params_ax_methods_avgs_adjusted_support10.csv")
df2 = df.drop(['c_angle_rule', 'u_angle_rule','c_tie_break'],axis=1).set_index("iters")
df2.to_csv("paper_bpp_loss_mse_loss_params_ax_methods_avgs_adjusted_support10.csv")
"""


from complexity.hypervolume_graphics import mohpo_grph



fldr = "/home/lucas/Documents/perceptronac/complexity/scripts/illustrations/"

y_axis_units="Log Hypervolume Difference"

col_lbls_map = {
    "sobol": "Sobol", 
    "ehvi": "qNEHVI",
    "parego": "qNParEGO",
    "c_gift_wrapping": "GLCH-CS",
    "u_gift_wrapping": "GLCH-US"}

for x_axis,title in zip(["params","flops"],["Number of Parameters","FLOPs"]):
    fil3 = f"paper_bpp_loss_mse_loss_{x_axis}_ax_methods_avgs_adjusted_support10.csv"    
    fig = mohpo_grph(fldr+fil3,title,False,None,None,y_axis_units,col_lbls_map)
    fig.savefig(fil3.replace(".csv",f"_{y_axis_units.replace(' ','_').lower()}.png"), dpi=300)




        

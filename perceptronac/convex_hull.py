# %%

import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from math import log10, floor

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))


def vnorm(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def vcos(v1,v2):
    return (v1[0]*v2[0] + v1[1]*v2[1]) / (vnorm(v1) * vnorm(v2))

def vdiff(v1,v2):
    return [v1[0]-v2[0],v1[1]-v2[1]]

def convex_hull(coord):

    c = min(coord,key=lambda x: x[0])
    ref = [0,-1] # vertical vector pointing down
    hull = [coord.index(c)]
    while ((hull[0] != hull[-1]) or (len(hull) == 1)):
        vcs=[vcos(vdiff(pt,c),ref) if pt != c else -math.inf for pt in coord]
        hull.append(vcs.index(max(vcs))) 
        p = coord[hull[-2]]
        c = coord[hull[-1]]
        ref = vdiff(c,p)
        if ref[1] >= 0: # horizontal vector pointing right
            hull.pop()
            break
    return hull


def points_in_convex_hull(data,x_col,y_col,log_x=False):
    """
    https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
    https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
    https://stackoverflow.com/questions/29188757/matplotlib-specify-format-of-floats-for-tick-labels
    https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation
    """


    data = data[[x_col,y_col]]
    normalized_data = data.copy()
    if log_x:
        normalized_data[x_col] = normalized_data[x_col].apply(lambda x : math.log10(x)) # complexity in logarithmic scale
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data.values) # make the coordinates in the range [0,1]

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(9.6,4.8), constrained_layout=True)

    ax[0].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x") # verify that everything is alright
    ax[1].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x") # verify that everything is alright

    chull = data.values[convex_hull(normalized_data.tolist()),:]
    ax[0].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x")
    ax[0].plot(chull[:,0],chull[:,1],linestyle="solid",color="red",marker=None)

    ax[1].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x")
    ax[1].plot(chull[:,0],chull[:,1],linestyle="solid",color="red",marker=None)


    if log_x:
        ax[0].set_xscale("log")
        xvalues = ax[0].get_xticks()
        ub = np.min(xvalues[xvalues>=np.max(data[x_col])])
        lb = np.max(xvalues[xvalues<=np.min(data[x_col])])
        xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]
        ax[0].set_xticklabels([])
        ax[0].set_xticklabels([], minor=True)
        ax[0].set_xticks(xvalues)
        ax[0].set_xticklabels(xvalues)

    if np.any((np.max(chull,axis=0) - np.min(chull,axis=0)) == 0):
        x_lb,x_ub = np.min(chull[:,0])-0.1*np.max(chull[:,0]),1.1*np.max(chull[:,0])
        y_lb,y_ub = np.min(chull[:,1])-0.1*np.max(chull[:,1]),1.1*np.max(chull[:,1])            
    else:
        x_range = np.max(chull[:,0]) - np.min(chull[:,0])
        y_range = np.max(chull[:,1]) - np.min(chull[:,1])
        x_lb,x_ub = np.min(chull[:,0])-0.1*x_range,np.max(chull[:,0])+0.1*x_range
        y_lb,y_ub = np.min(chull[:,1])-0.1*y_range,np.max(chull[:,1])+0.1*y_range

    ax[1].set_xlim(x_lb,x_ub)
    ax[1].set_ylim(y_lb,y_ub) 

    # xvalues = ax[1].get_xticks()
    # lb = np.min(xvalues[xvalues>=x_lb])
    # ub = np.max(xvalues[xvalues<=x_ub])
    # xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]

    xvalues = np.unique(
        [np.min(chull[:,0]),(np.max(chull[:,0])+np.min(chull[:,0]))/2,np.max(chull[:,0])])

    if len(xvalues) ==1 :
        xvalues = [x_lb,xvalues[0],x_ub]

    n_xticks = 3
    ax[1].set_xticks(xvalues[0:len(xvalues):np.max([len(xvalues)//n_xticks,1])])
    ax[1].set_xticklabels(xvalues[0:len(xvalues):np.max([len(xvalues)//n_xticks,1])])
    d = find_exp(np.min(np.diff(xvalues)))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter(f'%.{d+1}f'))

    ax[0].set_xlabel(x_col)
    ax[0].set_ylabel(y_col)
    ax[1].set_xlabel(x_col)
    ax[1].set_ylabel(y_col)

    return convex_hull(normalized_data.tolist()),fig


# %%

if __name__ == "__main__":

    data0 = pd.read_csv(
        "/home/lucas/Documents/perceptronac/results/exp_1656174158/exp_1656174158_train_values.csv")

    selected_points_mask0,fig0 = points_in_convex_hull(data0,"complexity","MLP",log_x=True)
    plt.show()

    print(data0.set_index("topology").iloc[selected_points_mask0,:])

    from perceptronac.loading_and_saving import plot_comparison

    data1 = pd.read_csv("/home/lucas/Documents/perceptronac/results/exp_1658197214/exp_1658197214_valid_values.csv")
    new_data1 = data1[np.logical_or(data1["quantization_bits"] > 4,data1["quantization_bits"] > 4)]
    new_data1.columns = ["data_bits/data_samples" if cn == "MLP" else cn for cn in new_data1.columns.values.tolist()]

    # fig1 = plot_comparison(new_data1["(data_bits+model_bits)/data_samples"].values,{"data_bits/data_samples":new_data1["data_bits/data_samples"].values},
    #         "(data_bits+model_bits)/data_samples",ylabel="data_bits/data_samples",xscale="log",
    #         linestyles={"data_bits/data_samples":"None"}, colors={"data_bits/data_samples":"k"}, markers={"data_bits/data_samples":"x"})

    selected_points_mask1,fig1= points_in_convex_hull(new_data1,
        "(data_bits+model_bits)/data_samples","data_bits/data_samples",log_x=True)
    
    plt.show()
    
    # fig1.savefig(f"/home/lucas/Desktop/output.png", dpi=300, facecolor='w', bbox_inches = "tight")

    print(new_data1.set_index("topology").iloc[selected_points_mask1,:])

# %%

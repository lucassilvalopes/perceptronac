import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


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

    data = data[[x_col,y_col]]
    normalized_data = data.copy()
    if log_x:
        normalized_data[x_col] = normalized_data[x_col].apply(lambda x : math.log10(x)) # complexity in logarithmic scale
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data.values) # make the coordinates in the range [0,1]

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4.8,4.8))

    ax.plot(data.values[:,0],data.values[:,1],linestyle="",marker="x") # verify that everything is alright

    chull = data.values[convex_hull(normalized_data.tolist()),:]
    ax.plot(data.values[:,0],data.values[:,1],linestyle="",marker="x")
    ax.plot(chull[:,0],chull[:,1],linestyle="solid",color="red",marker=None)

    if log_x:
        ax.set_xscale("log")
        xvalues = ax.get_xticks()
        ub = np.min(xvalues[xvalues>=np.max(data[x_col])])
        lb = np.max(xvalues[xvalues<=np.min(data[x_col])])
        xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]
        fig.axes[0].set_xticks(xvalues)
        fig.axes[0].set_xticklabels(xvalues)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    return convex_hull(normalized_data.tolist())
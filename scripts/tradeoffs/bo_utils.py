import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt


def simple_lambda_grid_3d():
    m45 = -45
    m90 = -90+1e-10
    m00 = -1e-10
    grid = -1/np.tan((np.pi/180) * np.array([
        [m45,m90,m90],
        [m45,m90,m45],
        [m45,m90,m00],
        [m45,m45,m90],
        [m45,m45,m45],
        [m45,m00,m90],
        [m45,m00,m00]
    ]))
    return grid


def plot_3d_lch(arrays_of_points,colors,markers,alphas,ax_ranges=None,ax_labels=None,title=None):
    """
    https://stackoverflow.com/questions/4739360/any-easy-way-to-plot-a-3d-scatter-in-python-that-i-can-rotate-around
    https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
    https://stackoverflow.com/questions/4930524/how-can-i-set-the-matplotlib-backend
    https://stackoverflow.com/questions/12358312/keep-plotting-window-open-in-matplotlib
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for data,c,m,a in zip(arrays_of_points,colors,markers,alphas):
        xs = [row[0] for row in data]
        ys = [row[1] for row in data]
        zs = [row[2] for row in data]

        ax.scatter(xs, ys, zs, c=c, marker=m, alpha=a)

    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_zlabel(ax_labels[2])
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    if ax_ranges is not None:
        ax.set_xlim(ax_ranges[0][0],ax_ranges[0][1])
        ax.set_ylim(ax_ranges[1][0],ax_ranges[1][1])
        ax.set_zlim(ax_ranges[2][0],ax_ranges[2][1])

    if title is None:
        plt.show(block=True)
    else:
        fig.savefig(
            f"teste3d.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


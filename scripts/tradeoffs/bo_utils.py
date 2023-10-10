
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


def plot_3d_lch(arrays_of_points,colors,markers):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for data,c,m in zip(arrays_of_points,colors,markers):
        xs = [row[0] for row in data]
        ys = [row[1] for row in data]
        zs = [row[2] for row in data]

        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.show()

    fig.savefig(
        f"teste3d.png", 
        dpi=300, facecolor='w', bbox_inches = "tight")


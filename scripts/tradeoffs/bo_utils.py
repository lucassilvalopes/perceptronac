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


def lambda_seq(ticks,angle_start=None,angle_end=None):
    m90 = -90+1e-10
    m00 = -1e-10
    if angle_start is None:
        angle_start = m00
    if angle_end is None:
        angle_end = m90
    ls = -1/np.tan((np.pi/180) * np.linspace(angle_start,angle_end,ticks))
    return ls


def lambda_grid_3d(y_lmbd,z_lmbd):
    zv,yv = np.meshgrid(z_lmbd, y_lmbd)
    zv_flat = zv.reshape(-1,1)
    yv_flat = yv.reshape(-1,1)
    xv_flat = np.ones((len(y_lmbd)*len(z_lmbd),1))
    grid = np.hstack([xv_flat,yv_flat,zv_flat])
    return grid


def plot_3d_lch(arrays_of_points,colors,markers,alphas,ax_ranges=None,ax_labels=None,title=None,planes=[]):
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

    for plane in planes:
        plot_plane_3d(ax,plane)

    if title is None:
        plt.show(block=True)
    else:
        fig.savefig(
            f"teste3d.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


def plane_coeff_from_pt_and_normal(point,normal):
    D = -np.dot(point,normal)
    A,B,C = normal[0],normal[1],normal[2]
    return (A,B,C,D)


def line_coeff_from_pts(x1,x2,y1,y2):
    a = (y2-y1) / (x2-x1)
    b = y1 - a * x1
    A = -a
    B = 1
    C = -b
    return (A,B,C) 


def line_coeff_from_pt_and_normal(point,normal):
    C = -np.dot(point,normal)
    A,B = normal[0],normal[1]
    return (A,B,C)


def plot_plane_3d(ax,plane):

    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    z_range = ax.get_zlim()

    all_lpts = get_points(x_range,y_range,z_range,plane)

    for lpts in all_lpts:
        ax.plot([e[0] for e in lpts], [e[1] for e in lpts],zs=[e[2] for e in lpts])



def get_points(x_range,y_range,z_range,plane):

    point = plane["center"]
    normal = plane["weights"]

    point = np.array(point).astype(np.float64)
    normal = np.array(normal).astype(np.float64)
    
    side_normals = np.array([
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1]
    ])

    side_points = np.array([
        [x_range[0],y_range[0],z_range[0]],
        [x_range[1],y_range[0],z_range[0]],
        [x_range[0],y_range[0],z_range[0]],
        [x_range[0],y_range[1],z_range[0]],
        [x_range[0],y_range[0],z_range[0]],
        [x_range[0],y_range[0],z_range[1]]
    ])

    boundaries_normals = np.array([
        [[1,0],[1,0],[0,1],[0,1]],
        [[1,0],[1,0],[0,1],[0,1]],
        [[1,0],[1,0],[0,1],[0,1]],
        [[1,0],[1,0],[0,1],[0,1]],
        [[1,0],[1,0],[0,1],[0,1]],
        [[1,0],[1,0],[0,1],[0,1]]
    ])

    boundaries_points = np.array([
        [[y_range[0],z_range[0]],[y_range[1],z_range[0]],[y_range[0],z_range[0]],[y_range[0],z_range[1]]],
        [[y_range[0],z_range[0]],[y_range[1],z_range[0]],[y_range[0],z_range[0]],[y_range[0],z_range[1]]],
        [[z_range[0],x_range[0]],[z_range[1],x_range[0]],[z_range[0],x_range[0]],[z_range[0],x_range[1]]],
        [[z_range[0],x_range[0]],[z_range[1],x_range[0]],[z_range[0],x_range[0]],[z_range[0],x_range[1]]],
        [[x_range[0],y_range[0]],[x_range[1],y_range[0]],[x_range[0],y_range[0]],[x_range[0],y_range[1]]],
        [[x_range[0],y_range[0]],[x_range[1],y_range[0]],[x_range[0],y_range[0]],[x_range[0],y_range[1]]]
    ])

    for p,n,ps,ns in zip(side_points,side_normals,boundaries_points,boundaries_normals):

        idx = np.argmax(n)

        pts = plane_intersect(
            plane_coeff_from_pt_and_normal(point,normal),
            plane_coeff_from_pt_and_normal(p,n),
        )
        if pts is None:
            continue

        pt1,pt2 = pts

        all_lpts = []

        if idx == 0:
            lpts = []
            for lp,ln in zip(ps,ns):
                pt = line_intersection(
                    line_coeff_from_pts(pt1[1],pt2[1],pt1[2],pt2[2]),
                    line_coeff_from_pt_and_normal(lp,ln)
                )
                if not (y_range[0] <= pt[0] <= y_range[1] and z_range[0] <= pt[1] <= z_range[1]):
                    pt = None
                if pt is None:
                    continue
                lpts.append([p[0],pt[0],pt[1]])
            assert len(lpts) == 2, lpts
            all_lpts.append(lpts)
            
        elif idx == 1:
            lpts = []
            for lp,ln in zip(ps,ns):
                pt = line_intersection(
                    line_coeff_from_pts(pt1[2],pt2[2],pt1[0],pt2[0]),
                    line_coeff_from_pt_and_normal(lp,ln)
                )
                if not (z_range[0] <= pt[0] <= z_range[1] and x_range[0] <= pt[1] <= x_range[1]):
                    pt = None
                if pt is None:
                    continue
                lpts.append([pt[1],p[1],pt[0]])
            assert len(lpts) == 2, lpts
            all_lpts.append(lpts)
        elif idx == 2:
            lpts = []
            for lp,ln in zip(ps,ns):
                pt = line_intersection(
                    line_coeff_from_pts(pt1[0],pt2[0],pt1[1],pt2[1]),
                    line_coeff_from_pt_and_normal(lp,ln)
                )
                if not (x_range[0] <= pt[0] <= x_range[1] and y_range[0] <= pt[1] <= y_range[1]):
                    pt = None
                if pt is None:
                    continue
                lpts.append([pt[0],pt[1],p[2]])
            assert len(lpts) == 2, lpts
            all_lpts.append(lpts)
            
    return all_lpts


def get_rotation_matrix(a,b):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    # a = np.array([1, 0, 0], dtype=np.float64)
    # b = np.array([0, 0, 1], dtype=np.float64)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1-c)/(s**2)
    return r


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)

    https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
    if np.linalg.det(A) == 0:
        return None

    p_inter = np.linalg.solve(A, d).T

    ret = (p_inter[0], (p_inter + aXb_vec)[0])

    return ret


def line_intersection(l1,l2):
    l1_vec, l2_vec = np.array(l1[:2]), np.array(l2[:2])
    A = np.array([l1_vec, l2_vec])
    d = np.array([-l1[2], -l2[2]]).reshape(2,1)
    if np.linalg.det(A) == 0:
        return None
    p_inter = np.linalg.solve(A, d).T
    return p_inter[0]


if __name__ == "__main__":

    import unittest

    class TestPlotPlane3D(unittest.TestCase):

        def test_utilitary_functions(self):
            point = np.ones((3,))
            normal = np.ones((3,))

            p = np.zeros((3,))
            n = np.array([0,0,1])

            pts = plane_intersect(
                plane_coeff_from_pt_and_normal(point,normal),
                plane_coeff_from_pt_and_normal(p,n),
            )

            pt1,pt2 = pts

            line_coeff = line_coeff_from_pts(pt1[0],pt2[0],pt1[1],pt2[1])

            self.assertTrue(np.allclose((1.0, 1, -3.0), line_coeff))

            boundaries_normals = np.array([[1,0],[1,0],[0,1],[0,1]])

            boundaries_points = np.array([[0,0],[4,0],[0,0],[0,4]])

            for i,lp,ln in zip(range(4),boundaries_points,boundaries_normals):
                pt = line_intersection(
                    line_coeff,
                    line_coeff_from_pt_and_normal(lp,ln)
                )
                if not (0 <= pt[0] <= 4 and 0 <= pt[1] <= 4):
                    pt = None
                if i == 0:
                    self.assertTrue(np.allclose(np.array([0., 3.]),pt))
                elif i == 1:
                    self.assertTrue(pt is None)
                elif i == 2:
                    self.assertTrue(np.allclose(np.array([3., 0.]),pt))
                elif i == 3:
                    self.assertTrue(pt is None)

    unittest.main()

    
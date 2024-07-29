
from complexity.glch_threed_utils import plane_intersect, plane_coeff_from_pt_and_normal
from complexity.glch_threed_utils import line_coeff_from_pts, line_intersection
from complexity.glch_threed_utils import line_coeff_from_pt_and_normal, get_points
import numpy as np
import unittest

class TestPlotPlane3D(unittest.TestCase):

    def setUp(self):
        self.point = np.ones((3,))
        self.normal = np.ones((3,))
        self.x_range = [0,4]
        self.y_range = [0,4]
        self.z_range = [0,4]

    def test_utilitary_functions(self):

        p = np.zeros((3,))
        n = np.array([0,0,1])

        pts = plane_intersect(
            plane_coeff_from_pt_and_normal(self.point,self.normal),
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

    def test_main_function(self):

        all_lpts = get_points(self.x_range,self.y_range,self.z_range,self.point,self.normal)

        print(all_lpts)


if __name__ == "__main__":
    unittest.main()

    
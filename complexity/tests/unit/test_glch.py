

import unittest
from complexity.glch import GLCHAngleRule


class TestGLCHAngleRule(unittest.TestCase):
    def setUp(self):
        self.ii = [0,2,3,7,8,9]
        self.deltacs = [450000,330000,270000,90000,566000,704000,101000,90000,613000,90000]
        self.deltars = [0.2,1.1,0.2,-0.01,2.1,0.003,-4.0,0.7,0.2,-1.1]
    def test_sorted_deltac_deltar_mode_0(self):
        sorted_idx = GLCHAngleRule.sorted_deltac_deltar(self.ii,self.deltacs,self.deltars,mode=0)
        sorted_idx == [9,3,7,2,0,8]
    def test_sorted_deltac_deltar_mode_1(self):
        sorted_idx = GLCHAngleRule.sorted_deltac_deltar(self.ii,self.deltacs,self.deltars,mode=1)
        sorted_idx == [9,3,8,0,2,7]


if __name__ == "__main__":
    unittest.main()
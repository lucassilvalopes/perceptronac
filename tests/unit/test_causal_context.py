import numpy as np
from perceptronac.coding2d import causal_context
import os
import unittest


class TestCausalContext(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"test_data")
        self.yt = np.load(os.path.join(test_data_dir,"yt.npz"))["arr_0"] # original vector of outcomes obtained using matlab
        self.Xt = np.load(os.path.join(test_data_dir,"Xt.npz"))["arr_0"] # original causal context matrix obtained using matlab
        self.imgtraining = np.load(os.path.join(test_data_dir,"imgtraining.npz"))["arr_0"] # original binary image obtained using matlab

    def test_shape(self):
        yt2,Xt2 = causal_context((self.imgtraining > 0).astype(int), 60)
        self.assertEqual(self.imgtraining.shape , (1024,791) )
        n_samples = ( 1024-int(np.ceil(np.sqrt(60))) ) * ( 791 -2* int(np.ceil(np.sqrt(60))))
        self.assertEqual(Xt2.shape , (n_samples, 60) )
        self.assertEqual(yt2.shape , (n_samples,1) )
        self.assertTrue(np.allclose(self.yt,yt2))
        self.assertTrue(np.allclose(self.Xt,Xt2))


if __name__ == "__main__":

    tests = TestCausalContext()
    print("running setUp")
    tests.setUp()
    print("running test_shape")
    tests.test_shape()
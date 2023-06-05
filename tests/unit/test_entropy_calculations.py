import os
import numpy as np
import unittest
import torch

from perceptronac.coding2d import causal_context
from perceptronac.context_training import context_training
from perceptronac.context_coding import context_coding
# from perceptronac.losses import perfect_AC
# from perceptronac.losses import Log2BCELoss

class TestEntropyCalculations(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"test_data")
        self.imgtraining = np.load(os.path.join(test_data_dir,"imgtraining.npz"))["arr_0"]
        self.imgcoding = np.load(os.path.join(test_data_dir,"imgcoding.npz"))["arr_0"]
        self.yt,self.Xt = causal_context((self.imgtraining > 0).astype(int), 10)
        self.yc,self.Xc = causal_context((self.imgcoding > 0).astype(int), 10)

        # test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"test_data")
        # yy = np.genfromtxt(os.path.join(test_data_dir,"yy.txt"), delimiter=",").reshape(-1,1)
        # X = np.genfromtxt(os.path.join(test_data_dir,"X.txt"), delimiter=",")
        # self.yt = yy[:yy.shape[0]//2,:]
        # self.Xt = X[:X.shape[0]//2,:]
        # self.yc = yy[yy.shape[0]//2:,:]
        # self.Xc = X[X.shape[0]//2:,:]

    def pytorch_rate(self,p,t):

        return torch.nn.BCELoss(reduction='mean')(p,t)/torch.log(torch.tensor(2,dtype=t.dtype))

    def numpy_rate(self, t, p):
        """https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Loss.cpp#L254"""
        rate = np.mean( - np.maximum( np.log(p) , -100 ) * (t == 1).astype(int) - \
            np.maximum(np.log(1 - p), -100) * (t == 0).astype(int)) / np.log(2)
        return rate

    def test_entropy_calculations_float64(self):

        pp = context_training(self.Xt,self.yt)

        pp = context_coding(self.Xc,pp)

        rate_a = self.pytorch_rate(torch.tensor(pp,dtype=torch.float64),torch.tensor(self.yc,dtype=torch.float64)).item()

        rate_b = self.numpy_rate(self.yc,pp)

        print(rate_a,rate_b)

        self.assertTrue(np.allclose(rate_a,rate_b))

    def test_entropy_calculations_float32(self):

        pp = context_training(self.Xt,self.yt)

        pp = context_coding(self.Xc,pp)

        rate_a = self.pytorch_rate(torch.tensor(pp).float(),torch.tensor(self.yc).float()).item()

        rate_b = self.numpy_rate(self.yc,pp.astype(np.float32) )

        print(rate_a,rate_b)

        self.assertTrue(np.allclose(rate_a,rate_b))


if __name__ == "__main__":

    tests = TestEntropyCalculations()
    print("running setUp")
    tests.setUp()
    print("running test_entropy_calculations_float64")
    tests.test_entropy_calculations_float64()
    print("running test_entropy_calculations_float32")
    tests.test_entropy_calculations_float32()
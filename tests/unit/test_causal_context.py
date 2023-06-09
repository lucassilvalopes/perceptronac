import numpy as np
from perceptronac.coding2d import causal_context
import os
import unittest
from perceptronac.utils import causal_context_many_imgs_rgb
from PIL import Image
import urllib.request


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


class TestCausalContextRGB(unittest.TestCase):
    def test_causal_context_many_imgs_rgb(self):
        """https://stackoverflow.com/questions/22676/how-to-download-a-file-over-http"""

        fpath = "baboon.png"
        url = "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/baboon.png"
        urllib.request.urlretrieve(url, fpath)
        img = Image.open(fpath)
        img = np.array(img)

        next = img[1:,1:-1,:].reshape(-1,3)
        previous = img[1:,:-2,:].reshape(-1,3)

        y,X = causal_context_many_imgs_rgb([fpath],1,interleaved=False)

        self.assertTrue( np.allclose(next.reshape((511,510,3)),y.reshape((511,510,3),order="F")) )
        # plt.imshow(y.reshape((511,510,3),order="F"))

        self.assertTrue( np.allclose(previous.reshape((511,510,3)),X.reshape((511,510,3),order="F")) )
        # plt.imshow(X.reshape((511,510,3),order="F"))


if __name__ == "__main__":

    tests = TestCausalContext()
    print("running setUp")
    tests.setUp()
    print("running test_shape")
    tests.test_shape()
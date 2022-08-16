from perceptronac.utils import luma_transform
from PIL import Image
import numpy as np

def test_luma_transform():
    im_arr = np.array([
        [[20,62,22],[12,155,42],[56,101,166]],
        [[192,201,40],[220,80,190],[144,74,173]],
        [[142,249,126],[123,173,199],[27,23,9]]
    ],dtype=np.uint8)
    im_pil = Image.fromarray(im_arr)
    np.allclose( np.array(im_pil.convert('L')) , luma_transform(im_arr,axis=2,keepdims=True))
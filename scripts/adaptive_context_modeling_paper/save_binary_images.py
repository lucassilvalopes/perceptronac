"""
https://stackoverflow.com/questions/47290668/image-fromarray-just-produces-black-image
"""

import os
import sys
from PIL import Image
from perceptronac.utils import read_im2bw_otsu

if __name__ == "__main__":

    src_folder = sys.argv[1]
    dst_folder = sys.argv[2]

    for f in os.listdir(src_folder):
        im = read_im2bw_otsu(os.path.join(src_folder,f))
        Image.fromarray((im * 255).astype('uint8'), mode='L').save(os.path.join(dst_folder,f))



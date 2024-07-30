"""
https://stackoverflow.com/questions/9983263/how-to-crop-an-image-using-pil
"""

import os
import sys
from PIL import Image

if __name__ == "__main__":
    src_folder = sys.argv[1]
    dst_folder = sys.argv[2]

    for f in os.listdir(src_folder):
        im = Image.open(os.path.join(src_folder,f))
        w, h = im.size
        im.crop(
            (50, 80, 720, 975) # (left, upper, right, lower)-tuple
        ).save(os.path.join(dst_folder,f),"PNG") 
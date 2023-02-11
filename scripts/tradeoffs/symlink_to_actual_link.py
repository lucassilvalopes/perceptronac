
"""
https://stackoverflow.com/questions/3220755/
https://stackoverflow.com/questions/123198/how-to-copy-files

python3 symlink_to_actual_link.py /home/lucas/Documents/data/vimeo90k_img/test /home/lucas/Documents/data/vimeo90k_img_nosl/test

python3 symlink_to_actual_link.py /home/lucas/Documents/data/vimeo90k_img/train /home/lucas/Documents/data/vimeo90k_img_nosl/train
"""
import os
import sys
import shutil

if __name__ == "__main__":

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    for fname in os.listdir(src_dir):
        slink = os.path.join(src_dir,fname)
        pth = os.path.realpath(slink)
        shutil.copyfile(pth, os.path.join(dst_dir,fname))
        
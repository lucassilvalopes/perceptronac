import os
import sys
import numpy as np


def get_folder_leaves_os_walk(start_path):
    folder_leaves = []
    for root, dirs, _ in os.walk(start_path):
        if not dirs:  # If 'dirs' is empty, it's a folder leaf
            folder_leaves.append(root)
    return folder_leaves


if __name__ == "__main__":

    src = "/home/lucaslopes/vimeo_triplet"

    lst = get_folder_leaves_os_walk(src)
    for pth in lst:
        new_pth = pth.replace("vimeo_triplet","vimeo_singlet")
        os.makedirs(new_pth, exist_ok=True)
        os.rename(os.path.join(pth,"im1.png"), os.path.join(new_pth,"im1.png"))
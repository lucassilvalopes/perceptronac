
import numpy as np
from PIL import Image
from netpbmfile import imwrite
import os
import subprocess as sb
from perceptronac.coding2d import causal_context
import perceptronac.coding3d as c3d


def read_im2bw(file_name,level):
    """
    Function with equivalent behaviour to matlab's im2bw
    
    Args:
        file_name : path to the image file
        level : threshold
    
    Returns:
        : numpy array representing the binary image
    """

    img = Image.open(file_name)
    
    thresh = level*255
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    return np.array(r).astype(int)


def read_im2gray(file_name):
    return np.array(Image.open(file_name).convert('L'))


def read_im2rgb(file_name):
    return np.array(Image.open(file_name))


def save_pbm(file_name, binary_image):
    """
    
    Args:
        binary_image : numpy array representing the binary_image
    """
    
    data = binary_image.astype(np.uint8)
    imwrite(file_name, data,maxval=1)


def im2pbm(im_path,pbm_path,th = 0.4):
    im = read_im2bw(im_path,th)
    save_pbm(pbm_path, im)
    return im.shape[:2]


def add_border(img,N):
    ns = int(np.ceil(np.sqrt(N)))
    nr,nc = img.shape[:2]
    new_img = 255*np.ones((nr+ns,nc+2*ns))
    new_img[ns:nr+ns,ns:nc+2*ns-ns] = img.copy()
    return new_img


def causal_context_many_imgs(pths,N,color_mode="binary"):
    if color_mode == "binary":
        return causal_context_many_imgs_binary(pths,N)
    elif color_mode == "gray":
        return causal_context_many_imgs_gray(pths,N)
    elif color_mode == "rgb":
        return causal_context_many_imgs_rgb(pths,N)
    else:
        raise ValueError(f"Color mode {color_mode} not supported. Options: binary, gray, rgb.")


def causal_context_many_imgs_binary(pths,N):
    y = []
    X = []
    for pth in pths:
        img = add_border(read_im2bw(pth,0.4),N)
        partial_y,partial_X = causal_context((img > 0).astype(int), N)
        y.append(partial_y)
        X.append(partial_X)
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X

def causal_context_many_imgs_gray(pths,N):
    y = []
    X = []
    for pth in pths:
        img = read_im2gray(pth)
        partial_y,partial_X = causal_context(img, N)
        y.append(partial_y)
        X.append(partial_X)
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X

def causal_context_many_imgs_rgb(pths,N,interleaved=True):
    y = []
    X = []
    for pth in pths:
        img = read_im2rgb(pth)
        partial_y = []
        partial_X = []
        for ch in range(3):
            ch_y,ch_X = causal_context(img[:,:,ch], N)
            partial_y.append(ch_y)
            if interleaved and N > 0:
                partial_X.append(np.expand_dims(ch_X,2))
            else:
                partial_X.append(ch_X)
        partial_y = np.concatenate(partial_y,axis=1)
        if interleaved and N > 0:
            partial_X = np.concatenate(partial_X,axis=2).reshape(-1,3*N, order='C')
        else:
            partial_X = np.concatenate(partial_X,axis=1)
        y.append(partial_y)
        X.append(partial_X)
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X


def causal_context_many_pcs(pths,N,percentage_of_uncles):
    y = []
    X = []

    M = int(percentage_of_uncles * N)
    print(f"using {N-M} siblings and {M} uncles.")
    for pth in pths:
        pc = c3d.read_PC(pth)[1]
        _,partial_X,partial_y,_,_ = c3d.pc_causal_context(pc, N-M, M)
        y.append(np.expand_dims(partial_y.astype(int),1) )
        X.append(partial_X.astype(int))
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X


def jbig1_rate(im_path):
    
#     src_path = "ieee_tip2017_klt1024_3.pbm"
#     dst_path = "ieee_tip2017_klt1024_3.jbg"

    src_path = "/tmp/tmp.pbm"
    dst_path = "/tmp/tmp.jbg"
    
    h,w = im2pbm(im_path,src_path,0.4)
    sb.run(["pbmtojbg","-q",src_path,dst_path])
    sz = os.path.getsize(dst_path) # bytes

    rate = 8 * sz / (w * h)
    
    os.remove(src_path)
    os.remove(dst_path)
    
    return rate

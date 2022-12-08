
import numpy as np
import inspect
from PIL import Image
from skimage import filters
from netpbmfile import imwrite
import os
import subprocess as sb
from perceptronac.coding2d import causal_context
import perceptronac.coding3d as c3d



def read_im2bw_otsu(file_name):
    """
    https://stackoverflow.com/questions/59113520/python-image-pillow-how-to-make-the-background-more-white-of-images
    https://stackoverflow.com/questions/65075158/converting-pil-image-to-skimage
    """

    img = np.array(Image.open(file_name).convert('L'))
    threshold = filters.threshold_otsu(img)
    result = (img>threshold).astype(int)
    return result


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


def im2pbm(im_path,pbm_path):
    im = read_im2bw_otsu(im_path)
    save_pbm(pbm_path, im)
    return im.shape[:2]


def add_border(img,N):
    ns = int(np.ceil(np.sqrt(N)))
    nr,nc = img.shape[:2]
    new_img = 255*np.ones((nr+ns,nc+2*ns))
    new_img[ns:nr+ns,ns:nc+2*ns-ns] = img.copy()
    return new_img


def causal_context_many_imgs(pths,N,n_classes=2,channels=[1,0,0],color_space="YCbCr"):
    if n_classes == 2 and channels==[1,0,0] and color_space == "YCbCr":
        return causal_context_many_imgs_binary(pths,N)
    elif n_classes == 256 and channels==[1,0,0] and color_space == "YCbCr":
        return causal_context_many_imgs_gray(pths,N)
    elif n_classes == 256 and channels==[1,1,1] and color_space == "RGB":
        return causal_context_many_imgs_rgb(pths,N)
    else:
        # https://stackoverflow.com/questions/582056/
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        m =  "Unsupported combination "+" ".join([f"{i}={values[i]}" for i in args])
        raise ValueError(m)


def causal_context_many_imgs_binary(pths,N):
    y = []
    X = []
    for pth in pths:
        img = add_border(read_im2bw_otsu(pth),N)
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


def causal_context_many_pcs(pths,N,percentage_of_uncles,geo_or_attr="geometry",n_classes=256,channels=[1,0,0],color_space="YCbCr"):
    if geo_or_attr == "geometry":
        return causal_context_many_pcs_geometry(pths,N,percentage_of_uncles)
    elif geo_or_attr == "attributes":
        if n_classes == 256 and channels==[1,0,0] and color_space == "YCbCr":
            return causal_context_many_pcs_gray(pths,N,percentage_of_uncles)
        elif n_classes == 256 and channels==[1,1,1] and color_space == "RGB":
            return causal_context_many_pcs_rgb(pths,N,percentage_of_uncles)
        else:
            raise ValueError("The specified combination of parameters for point cloud attributes is not supported yet.")
    else:
        raise ValueError(f"Unknown option {geo_or_attr}. Known options: geometry, attributes.")


def luma_transform(rgb,axis,keepdims):
    """
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
    """

    dimensions = list(rgb.shape)
    assert dimensions[axis]==3
    dimensions[axis]=1
    L = np.sum( 
        rgb * np.concatenate( 
            [np.ones(dimensions) * 299/1000 , np.ones(dimensions) * 587/1000 , np.ones(dimensions) * 114/1000],
            axis=axis 
        ),
        axis=axis, keepdims=keepdims
    )
    return L

def causal_context_many_pcs_gray(pths,N,percentage_of_uncles):

    y = []
    X = []

    M = int(percentage_of_uncles * N)
    print(f"using {N-M} siblings and {M} uncles.")
    for pth in pths:
        V,C = c3d.read_PC(pth)[1:]
        _,_,occupancy,_,_,partial_y,partial_X= c3d.pc_causal_context(V,N-M,M,C=C)
        y.append(luma_transform(partial_y[occupancy,:],axis=1,keepdims=True).astype(int))
        X.append(luma_transform(partial_X[occupancy,:,:],axis=2,keepdims=False).astype(int))
    y = np.concatenate(y,axis=0)
    X = np.concatenate(X,axis=0)
    return y,X


def causal_context_many_pcs_rgb(pths,N,percentage_of_uncles):
    y = []
    X = []

    M = int(percentage_of_uncles * N)
    print(f"using {N-M} siblings and {M} uncles.")
    for pth in pths:
        V,C = c3d.read_PC(pth)[1:]
        _,_,occupancy,_,_,partial_y,partial_X= c3d.pc_causal_context(V,N-M,M,C=C)
        y.append(partial_y[occupancy,:].astype(int))
        X.append(partial_X[occupancy,:,:].reshape(-1,3*N).astype(int))
    y = np.concatenate(y,axis=0)
    X = np.concatenate(X,axis=0)
    return y,X


def causal_context_many_pcs_geometry(pths,N,percentage_of_uncles):
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
    
    h,w = im2pbm(im_path,src_path)
    sb.run(["pbmtojbg","-q",src_path,dst_path])
    sz = os.path.getsize(dst_path) # bytes

    rate = 8 * sz / (w * h)
    
    os.remove(src_path)
    os.remove(dst_path)
    
    return rate

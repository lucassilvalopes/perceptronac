
import pandas as pd
import numpy as np
from collections import Counter
from attributes.losses import ac_lapl_rate


@np.vectorize
def matlab_round(x):
    return round(x)


def normalize_X(occupancy,wX,X):
    nX = np.where(occupancy!=-1,np.where(wX!=0,X/np.sqrt(wX),X),0)
    return nX


def quantize_y(y,Q):
    yq = matlab_round(y / Q)
    return yq


def read_data(config):
    wy = pd.read_csv(config["coeff_weights"],header=None,index_col=None).values
    y = pd.read_csv(config["coeff"],header=None,index_col=None).values[:,:1]
    X = pd.read_csv(config["neighbors"],header=None,index_col=None).values
    wX = pd.read_csv(config["neighbors_weights"],header=None,index_col=None).values
    occupancy = pd.read_csv(config["occupancy"],header=None,index_col=None).values
    
    nX = normalize_X(occupancy,wX,X)
    
    yq = quantize_y(y,config["Q"])
    
    return nX,yq,wy


def raht_lut(yq,wy,Q):
    """
    Implements the context modeling method described in
    `` Compression of 3D point clouds using a region-adaptive hierarchical transform,''
    The coefficients are assumed to have a laplacian distribution.
    The coefficients are grouped by weight (context) and the optimal standard deviation
    for the coefficients with that weight is computed.
    
    Params:
        yq : Nvox x 1, quantized coefficients
        wy : Nvox x 1, weights of the coefficients
        Q : scalar, quantization step
        
    Returns:
        sv: Nvox x 1, the optimal standard deviations of the quantized coefficients
    """
    
    Nms = Counter(wy.reshape(-1).tolist())
    subbands = sorted(Nms.keys()) 

    bms = []
    for sb in subbands:
        bms.append( np.sum(np.abs(yq[wy == sb]) * Q) / Nms[sb] )

    Nvox = yq.shape[0]
    sv = np.zeros((Nvox,))
    for n in range(Nvox):
        sv[n] = bms[subbands.index(wy[n])]

    sv = np.sqrt(2) * (sv / Q)
    
    return sv


def estimate_lut_rate(config):
    nX,yq,wy= read_data(config)
    rate = ac_lapl_rate(yq.reshape(-1),raht_lut(yq,wy,config["Q"]))/max(yq.shape)
    return rate


# function rate = perfect_AC(b, p)
# rate = mean(- log2(p) .* (b == 1) - log2(1 - p)  .* (b == 0));

import numpy as np
import torch

def perfect_AC(b, p, binary=True):
    if binary:
        return perfect_AC_binary(b, p)
    else:
        return perfect_AC_generic(b, p)


def perfect_AC_binary(b, p):
    """
    Args:
        b (-1 by 1): bitstream
        p (-1 by 1): probability of bit 1 as predicted by some method
    """
    rate = np.mean(
        - np.log2(p) * (b == 1).astype(int) - np.log2(1 - p)  * (b == 0).astype(int))
    return rate


def one_hot_encode(y, n_classes):
    one_hot = np.zeros((np.prod(y.shape),n_classes))
    for i in range(n_classes):
        vec = np.zeros((n_classes))
        vec[i] = 1
        one_hot[y.reshape(-1)==i] = vec
    new_shape = list(y.shape)
    new_shape.append(n_classes)
    return one_hot.reshape(*new_shape)


def perfect_AC_generic(b, p):
    """
    Args:
        b (-1 by n_channels): symbols to encode for each of the n_channels of each sample
        p (-1 by n_symbols by n_channels): probabilities for each of the n_symbols for each of the n_channels
    """
    rate = np.sum( - np.maximum( np.log(p) , -100) * np.transpose(one_hot_encode(b,p.shape[1]), (0, 2, 1)) ) / np.prod(b.shape)
    return rate/ np.log(2)


class Log2BCELoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.bce_loss(pred, target)/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class Log2CrossEntropyLoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.cross_entropy_loss(pred, target.long())/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class Log2NLLLoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.nll_loss(pred, target.long())/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


def ac_lapl_rate(xq, sd):
    """
    NUMBER OF BITS to transmit xq assuming xq[i] has a laplacian distribution with std sd[i]

    In the paper there was a Q in the exponent of the exponentials, 
    because the standard deviation was of dequantized coefficients.
    Here the standard deviation is of quantized coefficients. 
    If the standard deviation of quantized coefficients is sd, 
    then the standard deviation of dequantized coefficients is Q*sd. 
    Replacing the standard deviation of dequantized coefficients by Q*sd
    in the paper, gives the formulas used here.

    The mean of the coefficients of any bin is 0, regardless of the standard deviation.
    If the standard deviation is 0, this means the coefficients in the bin are 0.
    If they are zero, they do not need to be sent, they can simply be discarded, ignored.
    """

    nz = sd > 1e-6 
    
    xqa = np.abs(xq[nz])
    
    sdnz = sd[nz]
    
    rgt0 = (1/np.log(2)) * ( (np.sqrt(2) * xqa) / sdnz) - np.log2( np.sinh(1/(np.sqrt(2) * sdnz) ) )
    
    r0 = -np.log2(1-np.exp(-1/(np.sqrt(2) * sdnz)))
    
    r = rgt0 * (xqa > 0).astype(int) + r0 * (xqa == 0).astype(int)
    
    rateac = np.sum(r)
    
    return rateac


class LaplacianRate(torch.nn.Module):
    def forward(self, pred, target):
        """
        NUMBER OF BITS to transmit target assuming target[i,j] has a laplacian distribution with std pred[i,j]

        In the paper ``Transform coding for point clouds using a Gaussian
        process model'' there was a Q in the exponent of the exponentials, 
        because the standard deviation was of dequantized coefficients.
        Here the standard deviation is of quantized coefficients. 
        If the standard deviation of quantized coefficients is pred, 
        then the standard deviation of dequantized coefficients is Q*pred. 
        Replacing the standard deviation of dequantized coefficients by Q*pred
        in the paper, gives the formulas used here.

        target : quantized coefficients
        pred : standard deviation of quantized coefficients

        obs: negative standard deviation makes no sense
        """

        two = torch.tensor(2,dtype=target.dtype,device=target.device)
        
        rgt0 = (1/torch.log(two)) * ( (torch.sqrt(two) * torch.abs(target)) / pred) - torch.log2( torch.sinh(1/(torch.sqrt(two) * pred) ) )
        
        r0 = -torch.log2(1-torch.exp(-1/(torch.sqrt(two) * pred)))

        rgt0_mask = torch.abs(target) > 0

        r0_mask = torch.abs(target) == 0
        
        rateac = torch.sum(rgt0[rgt0_mask]) + torch.sum(r0[r0_mask])
        
        return rateac
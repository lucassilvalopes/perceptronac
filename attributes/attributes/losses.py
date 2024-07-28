
import numpy as np
import torch


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

import torch


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


class LaplacianRate(torch.nn.Module):
    def forward(self, pred, target):
        """
        In the paper ``Transform coding for point clouds using a Gaussian
        process model'' there was a Q in the exponent of the exponentials, 
        because the standard deviation was of dequantized coefficients.
        Here the standard deviation is of quantized coefficients. 
        If the standard deviation of quantized coefficients is sd, 
        then the standard deviation of dequantized coefficients is Q*sd. 
        Replacing the standard deviation of dequantized coefficients by Q*sd
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
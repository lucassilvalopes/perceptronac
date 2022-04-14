
# Lucas, do you have any plots of convergence (loss as a function of iteration number)?  
# If we make the batch size = 1, and plot the cross-entropy loss after each step k 
# (i.e., the code length for batch k = - y[k]log(q[k]) - (1-y[k])log(1-q[k]) where 
# q[k] is the output of the perceptron at step k), then this is exactly the code length 
# for batch k if we did backward-adaptive coding, like in CABAC.  We can also plot the 
# code length for CABAC after each step k in the same way.  (Or better, plot the average 
# code lengths up to step k, for both methods.  We should see this converge as O((log n)/n).)  
# I think this would greatly strengthen our story, as we could claim a drop-in replacement 
# for CABAC, with no pre-training required.  A related plot is Fig. 3 in your March 30 
# write-up, which shows the average bitrate after longer and longer training.

# The model gives you a prediction q, which gives you an error C
# analytically you have dC/dq and you evaluate at the current q
# backpropagation gives you dC/dw and dC/db for all w and b
# you walk one step in the direction opposite to dC/d* to minimize C
# if you use batch size of 1 you are updating in real time
# the learning rate will affect the convergence


import torch
import numpy as np
from perceptronac.models import MLP_N_64N_32N_1, CABAC
from perceptronac.utils import causal_context_many_imgs
from perceptronac.perfect_AC import perfect_AC
from perceptronac.models import Log2BCELoss, CausalContextDataset
from perceptronac.utils import plot_comparison
from perceptronac.utils import read_im2bw, save_values
from tqdm import tqdm


class RealTimeCABAC:

    def __init__(self,N):
        self.po2 = 2 ** np.arange(0,N).reshape(-1,1)
        self.c1 = np.zeros((2**N,1))
        self.c0 = np.zeros((2**N,1))
    def update(self,Xi,yi):
        context = Xi @ self.po2
        if (yi[0,0] == 1):
            self.c1[context[0,0],0] = self.c1[context[0,0],0] + 1
        else:
            self.c0[context[0,0],0] = self.c0[context[0,0],0] + 1
    def predict(self,Xi):
        context = Xi @ self.po2
        c1 = np.max([1,self.c1[context[0,0],0]])
        c0 = np.max([1,self.c0[context[0,0],0]])
        assert c1 > 0
        assert c0 > 0 
        pp = np.zeros((1,1)) 
        pp[0,0] = c1 / (c1 + c0)
        return pp

def backward_adaptive_coding(imgs,N,lr):
    
    device=torch.device("cuda:0")
    
    y,X = causal_context_many_imgs(imgs, N)
    # y0 = y[y.reshape(-1)==0,:]
    # X0 = X[y.reshape(-1)==0,:]
    # y1 = y[y.reshape(-1)==1,:]
    # X1 = X[y.reshape(-1)==1,:]

    # y = np.zeros((100,1))
    # X = np.zeros((100,X.shape[1]))

    # for i in range(50):
    #     y[2*i,:] = y0[i,:]
    #     X[2*i,:] = X0[i,:]
    # for i in range(50):
    #     y[2*i+1,:] = y1[i,:]
    #     X[2*i+1,:] = X1[i,:]

    trainset = CausalContextDataset(X,y)
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=1)
        
    model = MLP_N_64N_32N_1(N)
    model.to(device)
    model.train(True)

    cabac = RealTimeCABAC(N)

    criterion = Log2BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    mlp_avg_code_length_history = []
    cabac_avg_code_length_history = []

    mlp_running_loss = 0.0
    cabac_running_loss = 0.0
    iteration = 0

    for data in tqdm(dataloader):

        X_b,y_b= data
        X_b = X_b.float().to(device)
        y_b = y_b.float().to(device)

        optimizer.zero_grad()
        outputs = model(X_b)
        loss = criterion(outputs, y_b)
        loss.backward()
        optimizer.step()

        mlp_running_loss += loss.item()
        mlp_avg_code_length_history.append( mlp_running_loss / (iteration + 1) )
        # print("mlp out ",outputs.item()," mlp_loss ", loss.item(), "true ", y_b.item())
        assert y_b.item() == y[iteration,0]
        assert np.allclose(X_b.cpu().numpy().reshape(-1) , X[iteration,:].reshape(-1))

        cabac_pred_t = cabac.predict(X[iteration:iteration+1,:])
        cabac.update(X[iteration:iteration+1,:],y[iteration:iteration+1,:])

        cabac_loss = perfect_AC(y[iteration:iteration+1,:],cabac_pred_t)
        # print("cabac_pred_t ",cabac_pred_t," cabac_loss ", cabac_loss, "true ", y_b.item())
        cabac_running_loss += cabac_loss
        cabac_avg_code_length_history.append( cabac_running_loss / (iteration + 1) )

        # print(
        #     "iteration :" , iteration, 
        #     ", mlp avg code len :", mlp_running_loss / (iteration + 1), 
        #     # ", cabac avg code len :", cabac_running_loss / (iteration + 1)
        # )
        iteration += 1
            

    data = {
        "mlp": mlp_avg_code_length_history,
        "cabac": cabac_avg_code_length_history
    }
    
    return data

if __name__ == "__main__":

    img = read_im2bw("/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_3.png",0.4)

    for lr in (3.162277659**np.array([-2,-3,-4,-5,-6,-7,-9,-10,-11,-12,-13,-14,-15,-16])):

        data = backward_adaptive_coding([img],26,lr)
        
        len_data = len(data['mlp'])

        xvalues = np.arange( len_data )
        
        fig = plot_comparison(xvalues,data,"iter",
            mlp_marker = "", static_marker = "", cabac_marker = "")

        xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

        fig.axes[0].set_xticks( xticks)
        fig.axes[0].set_xticklabels( xticks)

        fname = f"backward_adaptive_coding_lr_{lr}".replace(".","pt")

        fig.savefig(fname+".png", dpi=300)

        save_values(fname,xvalues,data,"iter")
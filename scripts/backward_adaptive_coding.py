
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

import os
import torch
import numpy as np
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.utils import causal_context_many_imgs
from perceptronac.perfect_AC import perfect_AC
from perceptronac.models import Log2BCELoss, CausalContextDataset
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_values
from tqdm import tqdm


def weights_init(m):
    """
    https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156/2
    https://discuss.pytorch.org/t/access-weights-of-a-specific-module-in-nn-sequential/3627
    https://pytorch.org/docs/stable/nn.init.html
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight.data,0)
        torch.nn.init.constant_(m.bias.data,0)


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

def backward_adaptive_coding(pths,N,lr):
    
    device=torch.device("cuda:0")

    trainset = CausalContextDataset(pths,"image",N)
    y,X = trainset.y,trainset.X
    
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

    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=1)
        
    model = MLP_N_64N_32N_1(N)
    model.apply(weights_init)

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            weightvalue = set(model.layers[i].weight.detach().numpy().reshape(-1).tolist())
            assert len(weightvalue) == 1
            print(f"layer {i} weights properly initialized to {weightvalue}")
            biasvalue = set(model.layers[i].bias.detach().numpy().reshape(-1).tolist())
            assert len(biasvalue) == 1
            print(f"layer {i} biases properly initialized to {biasvalue}")
            

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
        "MLPlr={:.0e}".format(lr): mlp_avg_code_length_history,
        "LUT": cabac_avg_code_length_history
    }
    
    return data

if __name__ == "__main__":

    exp_name = "SPL2021_first_10_sorted_pages"

    pths = [os.path.join('SPL2021',f) for f in sorted(os.listdir('SPL2021'))[0:10]]

    learning_rates = (3.162277659**np.array([-2,-3,-4,-5,-6,-7,-8]))

    data = dict()
    for lr in learning_rates:
        data["MLPlr={:.0e}".format(lr)] = np.zeros((1024*768))
    data["LUT"] = np.zeros((1024*768))

    for pth in pths:
    
        for lr in learning_rates:

            partial_data = backward_adaptive_coding([pth],26,lr)
            for k in partial_data.keys():
                try:
                    data[k] = data[k] + np.array(partial_data[k])
                except Exception as e:
                    print(data)
                    raise e

    for k in data.keys():
        data[k] = data[k]/len(pths)
        
    len_data = len(data['LUT'])

    xvalues = np.arange( len_data )
        
    fig = plot_comparison(xvalues,data,"iteration",
        linestyles=8*["solid"],colors=["g","b","r","c","m","y","k","0.75"],markers=8*[""])

    xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

    fig.axes[0].set_xticks( xticks)
    fig.axes[0].set_xticklabels( xticks)

    fname = f"backward_adaptive_coding_{exp_name}"

    fig.savefig(fname+".png", dpi=300)

    save_values(fname,xvalues,data,"iteration")

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
from perceptronac.utils import read_im2bw
from tqdm import tqdm


def backward_adaptive_coding(imgs,N,lr):
    
    device=torch.device("cpu")
    
    y,X = causal_context_many_imgs(imgs, N)
    trainset = CausalContextDataset(X,y)
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=1)
        
    model = MLP_N_64N_32N_1(N)
    model.to(device)
    model.train(True)

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

        assert y_b.item() == y[iteration,0]
        assert np.allclose(X_b.numpy().reshape(-1) , X[iteration,:].reshape(-1))

        if iteration > 0:
            cabac = CABAC()
            cabac.fit(X[:iteration,:],y[:iteration,:])
            cabac_pred_t = cabac.predict(X[iteration:iteration+1,:],y[iteration:iteration+1,:])
            del cabac
        else:
            cabac_pred_t = 0.5
        cabac_running_loss += perfect_AC(y[iteration:iteration+1,:],cabac_pred_t)
        cabac_avg_code_length_history.append( cabac_running_loss / (iteration + 1) )

        print(
            "iteration :" , iteration, 
            ", mlp avg code len :", mlp_running_loss / (iteration + 1), 
            ", cabac avg code len :", cabac_running_loss / (iteration + 1)
        )
        iteration += 1
            

    data = {
        "mlp": mlp_avg_code_length_history,
        "cabac": cabac_avg_code_length_history
    }
    
    return data

if __name__ == "__main__":

    img = read_im2bw("/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_3.png",0.4)

    data = backward_adaptive_coding([img],26,0.0001)

    fig = plot_comparison(np.arange( len(data['mlp']) ),data,"iter")

    fig.savefig(f"backward_adaptive_coding.png", dpi=300)
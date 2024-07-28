

import torch
from attributes.losses import LaplacianRate

class LaplacianVarianceModel(torch.nn.Module):
    def __init__(self,N): 
        super().__init__()
        self.max_sd = 0

        a1_i = N
        a1_o = a2_i = min(8192,64*N) # min(2048,64*N)
        a2_o = a3_i = min(4096,32*N) # min(1024,32*N)
        a3_o = 1

        self.a1 = torch.nn.Linear( a1_i, a1_o )
        self.a1_act = torch.nn.ReLU()
        self.a2 = torch.nn.Linear( a2_i , a2_o )
        self.a2_act = torch.nn.ReLU()
        self.a3 = torch.nn.Linear( a3_i , a3_o )
        self.a3_act = torch.nn.Sigmoid()

        # self.b1 = torch.nn.Linear(N, min(1024,32*N) )
        # self.b1_act = torch.nn.ReLU()
        # self.b2 = torch.nn.Linear( min(1024,32*N), 1)
        # self.b2_act = torch.nn.ReLU()

    def forward(self, x):
        xa = self.a1(x)
        xa = self.a1_act(xa)
        xa = self.a2(xa)
        xa = self.a2_act(xa)
        xa = self.a3(xa)
        xa = self.a3_act(xa)

        # xb = self.b1(x)
        # xb = self.b1_act(xb)
        # xb = self.b2(xb)
        # xb = self.b2_act(xb)

        # return 0.01 + xa * (1 + xb )

        return 0.01 + self.max_sd * xa


class NNLaplacianVarianceModel:

    def __init__(self,configs,N):
        # seed = 7
        # torch.manual_seed(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        self.model = LaplacianVarianceModel(N)
        self.lr = configs["learning_rate"]
        self.batch_size = configs["batch_size"]

    def train(self,X,y):
        self.model.train()
        return self._apply(X,y,"train")

    def validate(self,X,y):
        self.model.eval()
        return self._apply(X,y,"valid")

    def _apply(self,X,y, phase):

        device = torch.device("cuda:0")

        model = self.model
        model.to(device)

        criterion = LaplacianRate()
        OptimizerClass=torch.optim.SGD
        optimizer = OptimizerClass(model.parameters(), lr=self.lr)

        dset = torch.utils.data.TensorDataset(X,y)
        dataloader = torch.utils.data.DataLoader(dset,batch_size=self.batch_size,shuffle=True)

        if phase == 'train':
            model.train(True)
        else:
            model.train(False) 

        running_loss = 0.0
        n_samples = 0.0

        # pbar = tqdm(total=np.ceil(len(dset)/self.batch_size))
        for data in dataloader:

            Xt_b,yt_b= data
            Xt_b = Xt_b.float().to(device)
            yt_b = yt_b.float().to(device)

            if phase == 'train':
                optimizer.zero_grad()
                model.max_sd = max([model.max_sd,torch.max(torch.abs(yt_b.detach())).item()])
                outputs = model(Xt_b)
                loss = criterion(outputs, yt_b)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(Xt_b)
                    loss = criterion(outputs, yt_b)

            running_loss += loss.item()
            n_samples += yt_b.numel()
        #     pbar.update(1)
        #     pbar.set_description(f"loss: {running_loss / n_samples} max_sd: {model.max_sd}")
        # pbar.close()

        return running_loss / n_samples , n_samples
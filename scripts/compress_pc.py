import torch
import numpy as np
from perceptronac.models import MLP_N_64N_32N_1, CausalContextDataset
import perceptronac.coding3d as c3d
from perceptronac.utils import causal_context_many_pcs
from perceptronac.adaptiveac.arithmetic_encoder import ArithmeticEncoder
from perceptronac.adaptiveac.arithmetic_decoder import ArithmeticDecoder
from perceptronac.adaptiveac.bitfile import BitFile
from perceptronac.adaptiveac.exceptions import WriteError
from tqdm import tqdm
from perceptronac.adaptiveac.utils import defineIntervals
import pandas as pd
from perceptronac.adaptiveac.exceptions import EndOfBinaryFile

class MockBitFile:
    def __init__(self,y):
        self.y = y
        self.i = 0
    def inputBits(self, bitCount):
        if self.i==len(self.y):
            raise EndOfBinaryFile
        r = self.convert(self.y[self.i:self.i+bitCount])
        self.i = self.i+bitCount
        return r 
    def convert(self,bitslist):
        integer = 0
        for i,b in enumerate(bitslist[::-1]):
            integer += b*(2**i)
        return integer
    def close(self):
        pass
    def reset(self):
        self.i = 0

if __name__ == "__main__":

    # 1) Carregar arquitetura e pesos.
    # Pra isso vai ser preciso especificar: 
    # A arquitetura, o N, e o caminho dos pesos
    #
    # 2) Carregar os dados a serem comprimidos
    # e obter as vizinhanças e os símbolos (X,y)
    # 
    # 3) Em um loop passar a prob e o símbolo
    # para o arithmetic_encoder
    #
    # OBS : é preciso comprimir todos os níveis

    pc = c3d.read_PC("/home/lucas/Desktop/computer_vision/mpeg-pcc-tmc13-v14.0/"+\
        "mpeg-pcc-tmc13-master/longdress/longdress_vox10_1051.ply")[1]
    last_level = 10

    # pc = c3d.read_PC("/home/lucas/Desktop/computer_vision/perceptronac/"+\
    #     "tests/test_data/vox1_test.ply")[1]
    # last_level = 1

    weights = "/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac_pytorch/"+\
    "results/exp_1649253745/exp_1649253745_047_model.pt"

    N = 47

    

    model = MLP_N_64N_32N_1(N)
    model.load_state_dict(torch.load(weights))
    model.train(False)

    pcs = [pc]
    for i in range(last_level-1,0,-1):
        pc = np.unique(np.floor(pc / 2),axis=0)
        pcs.append(pc)
    pcs = pcs[::-1]

    y,X = causal_context_many_pcs(pcs,N,0)

    dset = CausalContextDataset(X,y)

    dataloader = torch.utils.data.DataLoader(dset,batch_size=1,shuffle=False)

    encoderInputFile = MockBitFile(y.reshape(-1).astype(int).tolist())
    encoderOutputFile = BitFile("encoder_out", "wb")

    for v in tqdm(y):
        assert v == encoderInputFile.inputBits(1)
    try:
        encoderInputFile.inputBits(1)
        raise AssertionError
    except EndOfBinaryFile:
        pass
    encoderInputFile.reset()

    enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

    # print("writing encoder_out")
    # p = []
    # v = []
    # for data in tqdm(dataloader):
    #     X_b,y_b = data
    #     X_b = X_b.float() #.to(device)
    #     y_b = y_b.float() #.to(device)
    #     outputs = model(X_b)
    #     p.append(outputs.item())
    #     v.append(y_b.item())
    # #     counts = [
    # #         int(16000*(1-outputs.item())),
    # #         int(16000*outputs.item()),
    # #         1
    # #     ]
    # #     _,totals =defineIntervals(counts)
    # #     enc.do_one_step(totals)
    
    # # _,totals =defineIntervals(counts)
    # # done = enc.do_one_step(totals)
    # # assert done == 1

    # assert np.allclose(
    #     np.array(v).reshape(-1).astype(int),y.reshape(-1).astype(int))

    # df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
    # df.to_csv("data_to_encode.csv",index=False)

    df = pd.read_csv("/home/lucas/Desktop/computer_vision/perceptronac/tests/test_data/data_to_encode.csv")
    # df = pd.read_csv("data_to_encode.csv")

    probability_of_1 = df['probability_of_1'].values.tolist()
    bitstream = df['bitstream'].values.tolist()

    assert len(probability_of_1) == len(y)

    for p,v1,v2 in tqdm(list(zip(probability_of_1,bitstream,y.reshape(-1).tolist())) ):
        assert v1 == v2
        counts = [
            max(1,int(16000*(1-p))),
            max(1,int(16000*p)),
            1
        ]
        _,totals =defineIntervals(counts)
        done = enc.do_one_step(totals)
        assert done == 0
    
    _,totals =defineIntervals(counts)
    done = enc.do_one_step(totals)
    assert done == 1


    # del enc
    # del encoderInputFile
    # del encoderOutputFile

    # decoderInputFile = BitFile("encoder_out", "rb")
    # decoderOutputFile = BitFile("decoder_out", "wb")
    # dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)


    # # level 1
    # ptsL1 = c3d.xyz_displacements([0,1])
    # ptsL1 = ptsL1[np.lexsort((ptsL1[:, 2], ptsL1[:, 1], ptsL1[:, 0]))]

    # print("writing decoder_out")
    # for data in tqdm(dataloader):
    #     X_b,y_b = data
    #     X_b = X_b.float() #.to(device)
    #     y_b = y_b.float() #.to(device)
    #     outputs = model(X_b)
    #     counts = [
    #         int(16000*(1-outputs.item())),
    #         int(16000*outputs.item()),
    #         1
    #     ]
    #     _,totals =defineIntervals(counts)
    #     dec.do_one_step(totals)

    # _,totals =defineIntervals(counts)
    # done = dec.do_one_step(totals)
    # assert done == 1




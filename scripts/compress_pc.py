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
from perceptronac.coding3d import upsample_geometry

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

def lexsort(V):
    return V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]

if __name__ == "__main__":

    ############################ INPUTS ############################

    pc = c3d.read_PC("/home/lucas/Documents/data/longdress/longdress_vox10_1051.ply")[1]
    last_level = 10

    weights = "/home/lucas/Documents/results/exp_1649253745/exp_1649253745_047_model.pt"

    N = 47

    ############################# DATA #############################

    pcs = [pc]
    for i in range(last_level-1,0,-1):
        pc = np.unique(np.floor(pc / 2),axis=0)
        pcs.append(pc)
    pcs = pcs[::-1]

    y,X = causal_context_many_pcs(pcs,N,0)

    dset = CausalContextDataset(X,y)

    dataloader = torch.utils.data.DataLoader(dset,batch_size=1,shuffle=False)

    ############################# MODEL #############################

    model = MLP_N_64N_32N_1(N)
    model.load_state_dict(torch.load(weights))
    model.train(False)

    ################ WRITING PROBABILITIES AND DATA ##################

    print("writing data_to_encode.csv")
    p = []
    v = []
    for data in tqdm(dataloader):
        X_b,y_b = data
        X_b = X_b.float() #.to(device)
        y_b = y_b.float() #.to(device)
        outputs = model(X_b)
        p.append(outputs.item())
        v.append(y_b.item())

    assert np.allclose(
        np.array(v).reshape(-1).astype(int),y.reshape(-1).astype(int))

    df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
    df.to_csv("data_to_encode.csv",index=False)

    ################# READING PROBABILITIES AND DATA ##################

    df = pd.read_csv("data_to_encode.csv")

    probability_of_1 = df['probability_of_1'].values.tolist()
    bitstream = df['bitstream'].values.tolist()

    ################### WRITING ENCODER INPUT FILE ####################

    print("writing encoder_in")
    encoderInputFile = BitFile("encoder_in", "wb")
    for data in tqdm(dataloader):
        X_b,y_b = data
        encoderInputFile.outputBit(int(y_b.item()))
    
    ########### ENCODER INPUT FILE TO ENCODER OUTPUT FILE ##############

    encoderInputFile = BitFile("encoder_in", "rb")
    # encoderInputFile = MockBitFile(y.reshape(-1).astype(int).tolist())
    encoderOutputFile = BitFile("encoder_out", "wb")

    enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

    print("writing encoder_out")
    for p in tqdm(probability_of_1):
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

    del enc
    del encoderInputFile
    del encoderOutputFile

    ########### DECODER INPUT FILE TO DECODER OUTPUT FILE ##############

    decoderInputFile = BitFile("encoder_out", "rb")
    decoderOutputFile = BitFile("decoder_out", "wb")
    dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)

    print("writing decoder_out")
    for p in tqdm(probability_of_1):
        counts = [
            max(1,int(16000*(1-p))),
            max(1,int(16000*p)),
            1
        ]
        _,totals =defineIntervals(counts)
        dec.do_one_step(totals)

    _,totals =defineIntervals(counts)
    done = dec.do_one_step(totals)
    assert done == 1

    del dec
    del decoderInputFile
    del decoderOutputFile

    ################## FINAL POINT CLOUD RECONSTRUCTION ################

    decoderOutputFile = BitFile("decoder_out", "rb")
    decoderOutputFile.reset()

    last_level = 10

    for level in range(1,last_level+1):
        if level == 1:
            V_d = c3d.xyz_displacements([0,1])
        else:
            V_d = upsample_geometry(V_d, 2)
        V_d = lexsort(V_d)
        is_to_delete = []
        for i in range(len(V_d)):
            bit = decoderOutputFile.inputBits(1)
            if bit == 0:
                is_to_delete.append(i)
        V_d = np.delete(V_d,is_to_delete,axis=0)
        print(f"level {level} len : {len(V_d)}")

    pc = c3d.read_PC("/home/lucas/Documents/data/longdress/longdress_vox10_1051.ply")[1]
    assert np.allclose(lexsort(pc),lexsort(V_d))



    





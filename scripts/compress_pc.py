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

    # pc = c3d.read_PC("/home/lucas/Desktop/computer_vision/mpeg-pcc-tmc13-v14.0/"+\
    #     "mpeg-pcc-tmc13-master/longdress/longdress_vox10_1051.ply")[1]
    # last_level = 10

    pc = c3d.read_PC("/home/lucas/Desktop/computer_vision/perceptronac/"+\
        "tests/test_data/vox1_test.ply")[1]
    last_level = 1

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

    print("writing encoder_in")
    with open('encoder_in', 'wb') as f:
        encoderInputFile = BitFile("encoder_in", "wb")
        for data in tqdm(dataloader):
            X_b,y_b = data
            encoderInputFile.outputBit(int(y_b.item()))

    encoderInputFile = BitFile("encoder_in", "rb")
    encoderOutputFile = BitFile("encoder_out", "wb")

    enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

    print("writing encoder_out")
    p = []
    v = []
    for data in tqdm(dataloader):
        X_b,y_b = data
        X_b = X_b.float() #.to(device)
        y_b = y_b.float() #.to(device)
        outputs = model(X_b)
        p.append(outputs.item())
        v.append(y_b.item())
        counts = [
            int(16000*(1-outputs.item())),
            int(16000*outputs.item()),
            1
        ]
        _,totals =defineIntervals(counts)
        enc.do_one_step(totals)
    
    _,totals =defineIntervals(counts)
    done = enc.do_one_step(totals)
    assert done == 1

    assert np.allclose(
        np.array(v).reshape(-1).astype(int),y.reshape(-1).astype(int))

    df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
    df.to_csv("data_to_encode.csv",index=False)

    del enc
    del encoderInputFile
    del encoderOutputFile

    decoderInputFile = BitFile("encoder_out", "rb")
    decoderOutputFile = BitFile("decoder_out", "wb")
    dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)


    # level 1
    ptsL1 = c3d.xyz_displacements([0,1])
    ptsL1 = ptsL1[np.lexsort((ptsL1[:, 2], ptsL1[:, 1], ptsL1[:, 0]))]

    print("writing decoder_out")
    for data in tqdm(dataloader):
        X_b,y_b = data
        X_b = X_b.float() #.to(device)
        y_b = y_b.float() #.to(device)
        outputs = model(X_b)
        counts = [
            int(16000*(1-outputs.item())),
            int(16000*outputs.item()),
            1
        ]
        _,totals =defineIntervals(counts)
        dec.do_one_step(totals)

    _,totals =defineIntervals(counts)
    done = dec.do_one_step(totals)
    assert done == 1




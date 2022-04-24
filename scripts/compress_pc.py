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


def lexsort(V):
    return V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]


class MLP_N_64N_32N_1_PC_Coder:

    def __init__(self,weights,context_size,last_octree_level,pc_path):
        self.weights = weights
        self.N = context_size
        self.last_level = last_octree_level
        # TODO : remove the dependency of the decoder on the original pc
        self.pc_path = pc_path
        pc = c3d.read_PC(self.pc_path)[1]
        self.X,self.y = self.X_y_from_pc_pyramid(self.pc_pyramid(pc))
        
    def pc_pyramid(self,pc):
        pcs = [pc]
        for _ in range(self.last_level-1,0,-1):
            pc = np.unique(np.floor(pc / 2),axis=0)
            pcs.append(pc)
        pcs = pcs[::-1]
        return pcs

    def X_y_from_pc_pyramid(self,pcs):
        y = []
        X = []
        for pc in pcs:
            _,partial_X,partial_y,_,_ = c3d.pc_causal_context(pc, self.N, 0)
            y.append(np.expand_dims(partial_y.astype(int),1) )
            X.append(partial_X.astype(int))
        y = np.vstack(y)
        X = np.vstack(X)
        return X,y

    def p_y_from_X_y(self,X,y):

        model = MLP_N_64N_32N_1(self.N)
        model.load_state_dict(torch.load(self.weights))
        model.train(False)

        dset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        dataloader = torch.utils.data.DataLoader(dset,batch_size=1,shuffle=False)

        p = []
        v = []
        for data in tqdm(dataloader):
            X_b,y_b = data
            X_b = X_b.float() #.to(device)
            y_b = y_b.float() #.to(device)
            outputs = model(X_b)
            p.append(outputs.item())
            v.append(y_b.item())
        return p,v

    def store_p_y(self,p,v):
        df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
        df.to_csv("data_to_encode.csv",index=False)

    def load_p_y(self):
        df = pd.read_csv("data_to_encode.csv")
        probability_of_1 = df['probability_of_1'].values.tolist()
        bitstream = df['bitstream'].values.tolist()
        return probability_of_1,bitstream

    def write_encoder_inpt_file(self,y):
        print("writing encoder_in")
        encoderInputFile = BitFile("encoder_in", "wb")
        for y_b in y:
            encoderInputFile.outputBit(int(y_b))

    def batch_encode(self):
        # TODO : make pc_path an input to this method instead of an attribute of the class 
        # after removing the dependency of the decoder on the original pc

        # TODO: remove attributes and create a file with only the geometry

        p,v = self.p_y_from_X_y(self.X,self.y)

        self.store_p_y(p,v)
        p,v = self.load_p_y()

        self.write_encoder_inpt_file(v)

        self.encode(lambda x: x, p)

    def encode_realtime(self):
        # TODO : make pc_path an input to this method instead of an attribute of the class 
        # after removing the dependency of the decoder on the original pc

        # TODO: remove attributes and create a file with only the geometry

        # TODO: generate the context for the current sample only in the encoder

        self.write_encoder_inpt_file(self.y.reshape(-1).tolist())

        model = MLP_N_64N_32N_1(self.N)
        model.load_state_dict(torch.load(self.weights))
        model.train(False)

        dset = torch.utils.data.TensorDataset(torch.tensor(self.X))
        contextloader = torch.utils.data.DataLoader(dset,batch_size=1,shuffle=False)

        self.encode(lambda X_b: model(X_b.float()).item(), contextloader)

    def encode(self,predictor,contextloader):

        encoderInputFile = BitFile("encoder_in", "rb")
        encoderOutputFile = BitFile("encoder_out", "wb")
        enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

        # TODO : adapt this loop to be similar to the decoder loop

        print("writing encoder_out")
        for context in tqdm(contextloader):
            p = predictor(context)
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


    def batch_decode(self):

        predictor = lambda x: x
        context_generator = lambda pc,iteration : self.y[iteration,:]
        update_pc = lambda pc,symbol: pc
        self.decode(predictor,context_generator,update_pc)
        self.batch_reconstruct()


    def decode(self,predictor,context_generator,update_pc):

        decoderInputFile = BitFile("encoder_out", "rb")
        decoderOutputFile = BitFile("decoder_out", "wb")
        dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)

        print("writing decoder_out")
        pc = []
        iteration = 0
        while not (symbol == 2):
            p = predictor( context_generator(pc,iteration) )
            counts = [
                max(1,int(16000*(1-p))),
                max(1,int(16000*p)),
                1
            ]
            _,totals =defineIntervals(counts)
            symbol = dec.do_one_step(totals)
            pc = update_pc(pc, symbol)
            iteration += 1

        return pc


    def batch_reconstruct(self):

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

        pc = c3d.read_PC(self.pc_path)[1]
        assert np.allclose(lexsort(pc),lexsort(V_d))


if __name__ == "__main__":

    weights = "/home/lucaslopes/results/exp_1650138239/exp_1650138239_026_model.pt"
    context_size = 26
    last_octree_level = 10
    pc_path = "/home/lucaslopes/longdress/longdress_vox10_1300.ply"
    coder = MLP_N_64N_32N_1_PC_Coder(weights,context_size,last_octree_level,pc_path)
    coder.batch_encode()
    coder.batch_decode()
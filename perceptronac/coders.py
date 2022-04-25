import torch
import numpy as np
import itertools
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

    def __init__(self,weights,context_size,last_octree_level,mode = ""):
        self.weights = weights
        self.N = context_size
        self.last_level = last_octree_level
        self.mode = mode # "cache", "export", "" 

        self.data_to_encode = "data_to_encode.csv"
        self.encoder_in = "encoder_in"
        self.encoder_out = "encoder_out"
        self.decoder_out = "decoder_out"
        
    def _pc_pyramid(self,pc):
        pcs = [pc]
        for _ in range(self.last_level-1,0,-1):
            pc = np.unique(np.floor(pc / 2),axis=0)
            pcs.append(pc)
        pcs = pcs[::-1]
        return pcs

    def _X_y_from_pc_pyramid(self,pcs):
        y = []
        X = []
        for pc in pcs:
            _,partial_X,partial_y,_,_ = c3d.pc_causal_context(pc, self.N, 0)
            y.append(np.expand_dims(partial_y.astype(int),1) )
            X.append(partial_X.astype(int))
        y = np.vstack(y)
        X = np.vstack(X)
        return X,y

    def _p_y_from_X_y(self,X,y):
        if (self.mode == "cache"):
            return [],[]

        model = MLP_N_64N_32N_1(self.N)
        model.load_state_dict(torch.load(self.weights))
        model.train(False)

        dset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        dataloader = torch.utils.data.DataLoader(dset,batch_size=1,shuffle=False)

        p = []
        v = []
        for data in tqdm(dataloader, desc=f"writing {self.data_to_encode}"):
            X_b,y_b = data
            X_b = X_b.float() #.to(device)
            y_b = y_b.float() #.to(device)
            outputs = model(X_b)
            p.append(outputs.item())
            v.append(y_b.item())

        assert np.allclose(
            np.array(v).reshape(-1).astype(int),y.reshape(-1).astype(int))

        return p,v

    def _store_p_y(self,p,v):
        if (self.mode == "export"):
            df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
            df.to_csv(self.data_to_encode,index=False)
        elif (self.mode == "cache"):
            pass
        else:
            self.probability_of_1 = p
            self.bitstream = v

    def _load_p_y(self):
        if (self.mode == "export") or (self.mode == "cache"):
            df = pd.read_csv(self.data_to_encode)
            probability_of_1 = df['probability_of_1'].values.tolist()
            bitstream = df['bitstream'].values.tolist()
            return probability_of_1,bitstream
        else:
            return self.probability_of_1,self.bitstream

    def _write_encoder_inpt_file(self,y):
        encoderInputFile = BitFile(self.encoder_in, "wb")
        for y_b in tqdm(y, desc=f"writing {self.encoder_in}"):
            encoderInputFile.outputBit(int(y_b))

    def encode(self,pc_path):

        # TODO: remove attributes and create a file with only the geometry
        # to compare with the output from the decoder.

        pc = c3d.read_PC(pc_path)[1]

        X,y = self._X_y_from_pc_pyramid(self._pc_pyramid(pc))

        p,v = self._p_y_from_X_y(X,y)

        self._store_p_y(p,v)
        p,v = self._load_p_y()

        self._write_encoder_inpt_file(v)

        self._encode(lambda x: x, p)

        return self.encoder_out


    def _encode(self,predictor,contextloader):

        encoderInputFile = BitFile(self.encoder_in, "rb")
        encoderOutputFile = BitFile(self.encoder_out, "wb")
        enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

        for context in tqdm(contextloader,desc=f"writing {self.encoder_out}"):
            p = predictor(context)
            counts = [
                max(1,int(16000*(1-p))),
                max(1,int(16000*p)),
                1
            ]
            _,totals =defineIntervals(counts)
            symbol = enc.do_one_step(totals)
            assert symbol != 2

        _,totals =defineIntervals(counts)
        symbol = enc.do_one_step(totals)
        assert symbol == 2


    def decode(self):

        # TODO : make the decoder reconstruct the point cloud and predict the probabilitis from it 
        # in real time, removing the dependency on the vector of probabilities.

        # TODO : store the result in a file instead of returning the reconstructed point cloud
        # geometry.

        p,_ = self._load_p_y()
        predictor = lambda x: x
        context_generator = lambda pc,iteration : p[iteration,:]
        update_pc = lambda pc,symbol,iteration: pc
        _ = self._decode(predictor,context_generator,update_pc)
        return self._batch_reconstruct()


    def _decode(self,predictor,context_generator,update_pc):
        """
        https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
        https://stackoverflow.com/questions/5737196/is-there-an-expression-for-an-infinite-iterator
        """
        decoderInputFile = BitFile(self.encoder_out, "rb")
        decoderOutputFile = BitFile(self.decoder_out, "wb")
        dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)

        pc = []
        for iteration in tqdm(itertools.count(),desc=f"writing {self.decoder_out}"):
            p = predictor( context_generator(pc,iteration) )
            counts = [
                max(1,int(16000*(1-p))),
                max(1,int(16000*p)),
                1
            ]
            _,totals =defineIntervals(counts)
            symbol = dec.do_one_step(totals)
            if symbol == 2:
                break
            pc = update_pc(pc,symbol,iteration)

        return pc


    def _batch_reconstruct(self):

        decoderOutputFile = BitFile(self.decoder_out, "rb")
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

        return V_d


if __name__ == "__main__":

    weights = "/home/lucaslopes/results/exp_1649253745/exp_1649253745_047_model.pt"
    context_size = 47
    last_octree_level = 10
    pc_path = "/home/lucaslopes/longdress/longdress_vox10_1051.ply"
    coder = MLP_N_64N_32N_1_PC_Coder(weights,context_size,last_octree_level,mode = "cache")
    encoder_output_path = coder.encode(pc_path)
    recovered_pc = coder.decode(encoder_output_path)

    pc = c3d.read_PC(pc_path)[1]
    assert np.allclose(lexsort(pc),lexsort(recovered_pc))
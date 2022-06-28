import torch
import numpy as np
import math
import itertools
import os
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
from perceptronac.models import MLP_N_64N_32N_1_Constructor


def lexsort(V):
    return V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]


class PC_Coder:

    def __init__(self,model_constructor,context_size,last_octree_level,percentage_of_uncles=0):
        self._model_constructor = model_constructor
        self._last_level = last_octree_level
        self._M = int(percentage_of_uncles * context_size)
        self._N = context_size - self._M
        self._use_cache = False
        self._cache_path = None

    def enable_cache(self,cache_path = None):
        self._use_cache = True
        self._cache_path = cache_path

    def _setup_cache(self,pc_in):
        if self._use_cache and (self._cache_path is None):
            pc_id = os.path.splitext(os.path.basename(pc_in))[0]
            class_id = self.__class__.__name__
            self._cache_path = f"/tmp/{class_id}_{pc_id}_cache"
        
    def _pc_pyramid(self,pc):
        pcs = [pc]
        for _ in range(self._last_level-1,0,-1):
            pc = np.unique(np.floor(pc / 2),axis=0)
            pcs.append(pc)
        pcs = pcs[::-1]
        return pcs

    def _X_y_from_pc_pyramid(self,pcs):
        y = []
        X = []
        for pc in pcs:
            _,partial_X,partial_y,_,_ = c3d.pc_causal_context(pc, self._N, self._M)
            y.append(np.expand_dims(partial_y.astype(int),1) )
            X.append(partial_X.astype(int))
        y = np.vstack(y)
        X = np.vstack(X)
        return X,y

    def _p_y_from_X_y(self,X,y):

        model = self._model_constructor()

        batch_size = 2048

        dset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        dataloader = torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=False)

        p = []
        v = []
        for data in tqdm(dataloader, desc="preparing the data to encode"):
            X_b,y_b = data
            X_b = X_b.float() #.to(device)
            y_b = y_b.float() #.to(device)
            outputs = model(X_b)
            p.append(outputs.detach().cpu().numpy())
            v.append(y_b.detach().cpu().numpy())
        
        p = np.vstack(p)
        v = np.vstack(v)
        
        p=p.reshape(-1).tolist()
        v=v.reshape(-1).tolist()
        
        assert np.allclose(
            np.array(v).reshape(-1).astype(int),y.reshape(-1).astype(int))

        return p,v

    def _data_to_encode(self,pc):
        if self._use_cache and os.path.isfile(self._cache_path):
            return [],[]
        X,y = self._X_y_from_pc_pyramid(self._pc_pyramid(pc))
        p,v = self._p_y_from_X_y(X,y)

        return p,v

    def _store_p_y(self,p,v):
        if self._use_cache and os.path.isfile(self._cache_path):
            pass
        elif self._use_cache and not os.path.isfile(self._cache_path):
            df = pd.DataFrame(data = np.vstack([p,v]).T,columns=['probability_of_1','bitstream'])
            df.to_csv(self._cache_path,index=False)
        else:
            self._probability_of_1 = p
            self._bitstream = v

    def _load_p_y(self):
        if self._use_cache:
            df = pd.read_csv(self._cache_path)
            probability_of_1 = df['probability_of_1'].values.tolist()
            bitstream = df['bitstream'].values.tolist()
            return probability_of_1,bitstream
        else:
            return self._probability_of_1,self._bitstream

    def _write_encoder_inpt_file(self,y):
        encoderInputFile = BitFile(self._encoder_in, "wb")
        for y_b in tqdm(y, desc=f"writing {self._encoder_in}"):
            encoderInputFile.outputBit(int(y_b))

    def encode(self,pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out"):

        self._encoder_in = encoder_in
        self._encoder_out = encoder_out

        self._setup_cache(pc_in)

        pc = c3d.read_PC(pc_in)[1]

        p,v = self._data_to_encode(pc)

        self._store_p_y(p,v)
        p,v = self._load_p_y()

        self._write_encoder_inpt_file(v)

        self._encode(lambda x: x, p)


    def _encode(self,predictor,contextloader):

        encoderInputFile = BitFile(self._encoder_in, "rb")
        encoderOutputFile = BitFile(self._encoder_out, "wb")
        enc = ArithmeticEncoder(encoderInputFile, encoderOutputFile, 3, 1)

        for context in tqdm(contextloader,desc=f"writing {self._encoder_out}"):
            p = predictor(context)
            counts = [
                max(1,math.floor(16000*(1-p))),
                max(1,math.ceil(16000*p)),
                1
            ]
            _,totals =defineIntervals(counts)
            symbol = enc.do_one_step(totals)
            assert symbol != 2

        _,totals =defineIntervals(counts)
        symbol = enc.do_one_step(totals)
        assert symbol == 2


    def decode(self,pc_out,decoder_in="/tmp/encoder_out",decoder_out="/tmp/decoder_out"):

        # TODO : make the decoder reconstruct the point cloud and predict the probabilitis from it 
        # in real time, removing the dependency on the vector of probabilities.

        self._decoder_in = decoder_in
        self._decoder_out = decoder_out

        p,_ = self._load_p_y()
        predictor = lambda x: x
        context_generator = lambda pc,iteration : (p[iteration] if iteration < len(p) else p[-1])
        update_pc = lambda pc,symbol,iteration: pc
        _ = self._decode(predictor,context_generator,update_pc)
        xyz = self._batch_reconstruct()
        c3d.write_PC(pc_out,xyz)


    def _decode(self,predictor,context_generator,update_pc):
        """
        https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
        https://stackoverflow.com/questions/5737196/is-there-an-expression-for-an-infinite-iterator
        """
        decoderInputFile = BitFile(self._decoder_in, "rb")
        decoderOutputFile = BitFile(self._decoder_out, "wb")
        dec = ArithmeticDecoder(decoderInputFile, decoderOutputFile, 3, 1)

        pc = []
        for iteration in tqdm(itertools.count(),desc=f"writing {self._decoder_out}"):
            p = predictor( context_generator(pc,iteration) )
            counts = [
                max(1,math.floor(16000*(1-p))),
                max(1,math.ceil(16000*p)),
                1
            ]
            _,totals =defineIntervals(counts)
            symbol = dec.do_one_step(totals)
            if symbol == 2:
                break
            pc = update_pc(pc,symbol,iteration)

        return pc


    def _batch_reconstruct(self):

        decoderOutputFile = BitFile(self._decoder_out, "rb")
        decoderOutputFile.reset()

        last_level = 10

        for level in tqdm(range(1,last_level+1),desc="reconstructing the pc"):
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
            # print(f"level {level} len : {len(V_d)}")

        return V_d


def get_bpov(compressed_file,pc_len):
    return (os.path.getsize(compressed_file) * 8) / pc_len

if __name__ == "__main__":

    weights = "/home/lucaslopes/results/exp_1649253745/exp_1649253745_047_model.pt"
    context_size = 47
    last_octree_level = 10
    pc_path = "/home/lucaslopes/longdress/longdress_vox10_1051.ply"
    pc_in = pc_path.replace(".ply","_geo.ply")
    c3d.write_PC(pc_in,c3d.read_PC(pc_path)[1])
    pc_out= pc_in.replace(".ply","_rec.ply")
    cache =  None # "data_to_encode.csv"
    cache_enabled = False
    encoder_input_path = "encoder_in"
    encoder_output_path = "encoder_out"
    decoder_output_path = "decoder_out"

    print(
        f"""
        started coding
        weights: {weights}
        context_size: {context_size}
        last_octree_level: {last_octree_level}
        pc_in: {pc_in}
        pc_out: {pc_out}
        cache: {cache}
        cache_enabled: {cache_enabled}
        encoder_in: {encoder_input_path}
        encoder_out: {encoder_output_path}
        decoder_in: {encoder_output_path}
        decoder_out: {decoder_output_path}
        """, flush=True
    )

    constructor = MLP_N_64N_32N_1_Constructor(context_size,weights)
    coder = PC_Coder(constructor.construct,context_size,last_octree_level)
    if cache_enabled:
        print("enabling cache")
        coder.enable_cache(cache)
    coder.encode(pc_in,encoder_in=encoder_input_path,encoder_out=encoder_output_path)
    coder.decode(pc_out,decoder_in=encoder_output_path,decoder_out=decoder_output_path)

    pc = c3d.read_PC(pc_in)[1]
    recovered_pc = c3d.read_PC(pc_out)[1]
    assert np.allclose(lexsort(pc),lexsort(recovered_pc))
    print("\npoint cloud successfully reconstructed")
    print(f"bpov : {get_bpov(encoder_output_path,len(pc))}")
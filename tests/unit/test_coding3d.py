
import perceptronac.coding3d as c3d
from perceptronac.coding3d import mortoncode as mc
import numpy as np
import abc
import unittest


class TestPcCausalContext(unittest.TestCase):
    
    def setUp(self):
        # _, self.V, _ = c3d.read_PC(
        # "/home/lucas/Desktop/computer_vision/mpeg-pcc-tmc13-v14.0/mpeg-pcc-tmc13-master/longdress/longdress_vox10_1300.ply"
        # )
        self.V = np.unique(np.random.randint(0,2**8-1,size = (5500,3)),axis=0)
        self.N = 14
        self.M = 0
        self.V_nni,self.contexts,self.occupancy,self.this_nbhd,self.prev_nbhd=\
            c3d.pc_causal_context(self.V, self.N, self.M)
        
    def test_pc_causal_context_causality(self):
        assert np.all(self.this_nbhd[:,0] <=0)
        assert np.all(self.this_nbhd[self.this_nbhd[:,0]==0][:,1] <=0)
        assert np.all(self.this_nbhd[np.logical_and(
            self.this_nbhd[:,0]==0,self.this_nbhd[:,1]==0)][:,2] < 0)
    
    def test_no_column_equal_to_output(self):
        column_equal_to_output = False
        for i in range(self.contexts.shape[1]):
            column_equal_to_output = (column_equal_to_output or \
                bool(np.all(self.contexts[:,i] == self.occupancy)))
        assert (column_equal_to_output is False)

    def test_occupancy(self):
        self.assertTrue(
            np.allclose(
                lexsort(self.V_nni[self.occupancy]),
                lexsort(self.V)
            )
        )

    def test_points_existence(self):

        pts = []
        for i in range(self.V_nni.shape[0]):
            pt = self.V_nni[i:i+1,:]
            this_neighs=pt+self.this_nbhd[self.contexts[i][:self.N]]
            pts.append(this_neighs)
        this_neighs = np.vstack(pts)
        not_in_V= set(map(str,this_neighs.astype(int))) - set(map(str,self.V.astype(int)))
        not_in_V_cnt = len(not_in_V)
        self.assertTrue( not_in_V_cnt == 0, msg= f"{not_in_V}")

    def test_points_inexistence(self):
        not_contexts = np.logical_not(self.contexts)
        pts = []
        for i in range(self.V_nni.shape[0]):
            pt = self.V_nni[i:i+1,:]
            this_not_neighs=pt+self.this_nbhd[not_contexts[i][:self.N]]
            pts.append(this_not_neighs)
        this_not_neighs = np.unique(np.vstack(pts),axis=0)
        
        setA = set(map(str,self.V.astype(int)))
        setB = set(map(str,this_not_neighs.astype(int)))
        # print(setA)
        # print(setB)
        # print(setA.intersection(setB))
        self.assertEqual( len(setA.union(setB)), len(setA) + len(setB) )
        

def lexsort(V):
    return V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]

class TestPcCausalContextDensePc(unittest.TestCase):
    
    def setUp(self):
        x,y,z = np.meshgrid(range(10),range(10),range(10))
        self.V = np.vstack([x.reshape(-1),y.reshape(-1),z.reshape(-1)]).T
        self.N = 122
        self.M = 6
        self.V_nni,self.contexts,self.occupancy,self.this_nbhd,self.prev_nbhd=\
            c3d.pc_causal_context(self.V, self.N, self.M)
        
    def test_pc_causal_context_at_center_of_dense_pc(self):
        assert np.all(self.contexts[self.V_nni.tolist().index([5,5,5])])

    def test_points_existence_dense_pc(self):

        all_neighbors = ( np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0) ).transpose([0,2,1]).reshape(-1,3)

        new_V= np.unique(all_neighbors[self.contexts[:,:self.N].reshape(-1)],axis=0)

        new_V = np.vstack([ new_V,np.array([[9,9,9]]) ])

        self.assertEqual(len(new_V),len(self.V))
        self.assertTrue( np.allclose(lexsort(new_V),lexsort(self.V)) )


class TestPcCausalContextFirstQuadrant(unittest.TestCase):   
    
    def setUp(self):
        x,y,z = np.meshgrid(range(5),range(5),range(5))
        self.V = np.vstack([x.reshape(-1),y.reshape(-1),z.reshape(-1)]).T
        self.N = 122
        self.M = 6
        self.V_nni,self.contexts,self.occupancy,self.this_nbhd,self.prev_nbhd=\
            c3d.pc_causal_context(self.V, self.N, self.M)
        
    def test_first_quadrant_occupancy(self):
        siblings = np.array([[5,5,5]]) + self.this_nbhd[self.contexts[self.V_nni.tolist().index([5,5,5])][:self.N]]
        uncles = np.floor(np.array([[5,5,5]])/2) + self.prev_nbhd[self.contexts[self.V_nni.tolist().index([5,5,5])][self.N:]]
        assert np.all(siblings <= 5)
        assert np.all(uncles <= 2)
        


import perceptronac.coding3d as c3d
from perceptronac.coding3d import mortoncode as mc
import numpy as np
import abc
import unittest


class TestPcCausalContextRandomColoredPc(unittest.TestCase):
    

    # def setUp(self):
    #     pass

    @classmethod
    def setUpClass(cls):
        # https://stackoverflow.com/questions/14305941/run-setup-only-once-for-a-set-of-automated-tests
        super().setUpClass()
        cls.n_points = 5500
        cls.n_coordinates = 3
        cls.geometry_bits = 8 # vox 8
        cls.color_bits = 8 # rgb colors
        cls.n_channels = 3
        cls.V = np.unique(np.random.randint(0,2**cls.geometry_bits,size = (cls.n_points,cls.n_coordinates)),axis=0)
        cls.n_points = cls.V.shape[0] # updating n_points
        cls.C = np.random.randint(0,2**cls.color_bits,size = (cls.n_points,cls.n_channels))
        cls.ordering = 1
        cls.N = 14
        cls.M = 0
        cls.V,cls.C = c3d.sort_V_C(cls.V,cls.C,ordering=cls.ordering)
        cls.V_nni,cls.contexts,cls.occupancy,cls.this_nbhd,cls.prev_nbhd,cls.C_nni,cls.contexts_color=\
            c3d.pc_causal_context(cls.V, cls.N, cls.M,ordering=cls.ordering,C=cls.C)

    def test_V_nni_shape(self):

        self.assertTrue( self.V_nni.shape[0] == self.occupancy.shape[0])

    def test_C_nni_shape(self):

        self.assertTrue( self.C_nni.shape[0] == self.occupancy.shape[0])

    def test_contexts_shape(self):

        self.assertTrue( self.contexts.shape[0] == self.occupancy.shape[0] )
        self.assertTrue( self.contexts.shape[1] == self.N + self.M)

    def test_contexts_color_shape(self):

        self.assertTrue( self.contexts_color.shape[0] == self.occupancy.shape[0] )
        self.assertTrue( self.contexts_color.shape[1] == self.N + self.M)
        self.assertTrue( self.contexts_color.shape[2] == self.n_channels )

    def test_reconstruct_V(self):

        self.assertTrue(np.allclose(self.V,self.V_nni[self.occupancy] ) )

    def test_reconstruct_C(self):

        self.assertTrue(np.allclose(self.C,self.C_nni[self.occupancy] ) )


    def test_occupied_points_really_exist(self):
        all_points = np.transpose(np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0), (0, 2, 1)).reshape(-1,3)
        supposed_to_exist = np.unique(all_points[self.contexts[:,:self.N].reshape(-1)],axis=0)
        not_in_V= set(map(str,supposed_to_exist.astype(int).tolist())) - set(map(str,self.V.astype(int).tolist()))
        self.assertEqual( len(not_in_V), 0 )


    def test_not_occupied_points_really_do_not_exist(self):
        all_points = np.transpose(np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0), (0, 2, 1)).reshape(-1,3)
        supposed_to_not_exist = np.unique(all_points[np.logical_not(self.contexts[:,:self.N].reshape(-1))],axis=0)
        in_V= set(map(str,supposed_to_not_exist.astype(int).tolist())).intersection( set(map(str,self.V.astype(int).tolist())) )
        self.assertEqual( len(in_V), 0 )

# class TestPcCausalContextHandcraftedPc(unittest.TestCase):

#     def test_handcrafted_pc():


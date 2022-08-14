
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
        all_points = np.transpose(np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0), (0, 2, 1)).reshape(-1,self.n_coordinates)
        supposed_to_exist = np.unique(all_points[self.contexts[:,:self.N].reshape(-1)],axis=0)
        not_in_V= set(map(str,supposed_to_exist.astype(int).tolist())) - set(map(str,self.V.astype(int).tolist()))
        self.assertEqual( len(not_in_V), 0 )


    def test_not_occupied_points_really_do_not_exist(self):
        all_points = np.transpose(np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0), (0, 2, 1)).reshape(-1,self.n_coordinates)
        supposed_to_not_exist = np.unique(all_points[np.logical_not(self.contexts[:,:self.N].reshape(-1))],axis=0)
        in_V= set(map(str,supposed_to_not_exist.astype(int).tolist())).intersection( set(map(str,self.V.astype(int).tolist())) )
        self.assertEqual( len(in_V), 0 )


    def test_colors_exist_if_and_only_if_points_are_occupied(self):
        recovered_contexts_occupancy = np.logical_not( np.all(self.contexts_color == -1,axis=2) )
        self.assertEqual(len(recovered_contexts_occupancy.shape),len(self.contexts.shape))
        self.assertEqual(recovered_contexts_occupancy.shape[0],self.contexts.shape[0])
        self.assertEqual(recovered_contexts_occupancy.shape[1],self.contexts.shape[1])
        self.assertEqual( np.count_nonzero(recovered_contexts_occupancy) , np.count_nonzero(self.contexts) )
        self.assertTrue(np.allclose(recovered_contexts_occupancy.astype(int),self.contexts.astype(int)))


    def test_colors_are_correct(self):
        all_points = np.transpose(np.expand_dims(self.V_nni,2) + np.expand_dims(self.this_nbhd.T,0), (0, 2, 1))
        point_color_combinations = np.concatenate([all_points,self.contexts_color[:,:self.N,:]],axis=2).reshape(-1,self.n_coordinates+self.n_channels)
        point_color_combinations = np.unique(point_color_combinations[self.contexts[:,:self.N].reshape(-1)],axis=0)
        true_point_color_combinations = np.concatenate([self.V,self.C],axis=1)
        not_in_V= set(map(str,point_color_combinations.astype(int).tolist())) - set(map(str,true_point_color_combinations.astype(int).tolist()))
        assert len(not_in_V) == 0, f"{point_color_combinations[:10]} {true_point_color_combinations[:10]}"



class TestPcCausalContextHandcraftedPc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.n_coordinates = 3
        cls.n_channels = 3
        cls.n_points = 4

        cls.V = np.array([
            [0,0,0], # most distant
            [1,0,0], # medium distance
            [1,1,0], # closest
            [1,1,1] # center point
        ])
        cls.C = np.array([
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,255]
        ])
        cls.ordering = 1
        cls.N = 13
        cls.M = 1
        cls.V,cls.C = c3d.sort_V_C(cls.V,cls.C,ordering=cls.ordering)
        cls.V_nni,cls.contexts,cls.occupancy,cls.this_nbhd,cls.prev_nbhd,cls.C_nni,cls.contexts_color=\
            c3d.pc_causal_context(cls.V, cls.N, cls.M,ordering=cls.ordering,C=cls.C)

    def test_V_nni(self):

        expected_V_nni = np.array([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ]) 

        self.assertTrue(np.allclose(expected_V_nni,self.V_nni))

    def test_occupancy(self):

        expected_occupancy = np.array([True,False,False,False,True,False,True,True])

        self.assertTrue(np.allclose(expected_occupancy,self.occupancy))


    def test_C_nni(self):

        expected_occupancy = np.array([True,False,False,False,True,False,True,True])

        expected_C_nni = np.array([
            [255,0,0],
            [-1,-1,-1],
            [-1,-1,-1],
            [-1,-1,-1],
            [0,255,0],
            [-1,-1,-1],
            [0,0,255],
            [255,255,255],
        ]) 

        self.assertTrue(np.allclose(expected_C_nni[expected_occupancy],self.C_nni[self.occupancy]))


    def test_this_nbhd(self):

        expected_this_nbhd = np.array([ # sort first in causal raster XYZ order, then by increasing distance
            [-1,0,0],
            [0,-1,0],
            [0,0,-1],
            [-1,-1,0],
            [-1,0,-1],
            [-1,0,1],
            [-1,1,0],
            [0,-1,-1],
            [0,-1,1],
            [-1,-1,-1],
            [-1,-1,1],
            [-1,1,-1],
            [-1,1,1],
        ])  

        self.assertEqual(expected_this_nbhd.shape[0],self.this_nbhd.shape[0])
        self.assertTrue(len( set(map(str,expected_this_nbhd.astype(int).tolist())) - set(map(str,self.this_nbhd.astype(int).tolist())) ) == 0)
        assert np.allclose(expected_this_nbhd,self.this_nbhd) , f"{expected_this_nbhd} {self.this_nbhd}"

    def test_prev_nbhd(self):

        expected_prev_nbhd = np.array([
            [0,0,0],
        ])
        self.assertEqual(expected_prev_nbhd.shape[0],self.prev_nbhd.shape[0])
        self.assertTrue(len( set(map(str,expected_prev_nbhd.astype(int).tolist())) - set(map(str,self.prev_nbhd.astype(int).tolist())) ) == 0)
        assert np.allclose(expected_prev_nbhd,self.prev_nbhd) , f"{expected_prev_nbhd} {self.prev_nbhd}"

    def test_contexts(self):

        expected_contexts_a = np.array([
            [False,False,False,False,False,False,False,False,False,False,False,False,False],
            [False,False,True,False,False,False,False,False,False,False,False,False,False],
            [False,True,False,False,False,False,False,False,False,False,False,False,False],
            [False,False,False,False,False,False,False,True,False,False,False,False,False],
            [True,False,False,False,False,False,False,False,False,False,False,False,False],
            [False,False,True,False,True,False,False,False,False,False,False,False,False],
            [False,True,False,True,False,False,False,False,False,False,False,False,False],
            [False,False,True,False,False,False,False,True,False,True,False,False,False]
        ])

        expected_contexts_b = np.array([
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]
        ])

        expected_contexts = np.concatenate([expected_contexts_a,expected_contexts_b],axis=1)

        self.assertTrue(np.allclose(expected_contexts,self.contexts))

    def test_contexts_color(self):

        expected_contexts_color_a = np.array([
            [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[-1,-1,-1],[0,255,0],[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[0,255,0],[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
            [[-1,-1,-1],[-1,-1,-1],[0,0,255],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[0,255,0],[-1,-1,-1],[255,0,0],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
        ])

        expected_contexts_color_b = np.array([
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]],
            [[127.5,127.5,127.5]]
        ])

        expected_contexts_color = np.concatenate([expected_contexts_color_a,expected_contexts_color_b],axis=1)

        self.assertTrue(np.allclose(expected_contexts_color,self.contexts_color))

import numpy as np
from perceptronac.coding3d import mortoncode as mc
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import warnings


def xyz_displacements(interval):
    x, y, z = np.meshgrid(interval, interval, interval)
    deltas = np.array([x.flatten('F'), y.flatten('F'), z.flatten('F')]).T

    return deltas[np.lexsort((deltas[:, 2], deltas[:, 1], deltas[:, 0]))]


def upsample_geometry(V, s):
    deltas = xyz_displacements(np.arange(0, s, 1))
    up_delta = np.ones((deltas.shape[0], 1))
    up_VC = np.ones((V.shape[0], 1))
    Vup = np.kron(V * s, up_delta) + np.kron(up_VC, deltas)
    return Vup


def ismember_xyz(V1, V2):
    hash1 = mc.morton_code(V1)
    hash2 = mc.morton_code(V2)
    return np.isin(hash1, hash2)


def read_PC(path):
    pc = o3d.io.read_point_cloud(path)
    V = np.asarray(pc.points)
    C = np.asarray(pc.colors) * 255
    # N = ?
    return pc, V, C


def write_PC(filename,xyz):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(filename, pointcloud, write_ascii=True)


def parents_children(V, ordering):
    """
    V : input V sorted according to input ordering
    V_d : parents
    V_nni : parents' children sorted according to input ordering
    """
    if ordering == 1:
        V = V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]
    elif ordering == 2:
        V = V[np.lexsort((V[:, 0], V[:, 2], V[:, 1]))]
    elif ordering == 3:
        V = V[np.lexsort((V[:, 1], V[:, 0], V[:, 2]))]

    V_d = np.unique(np.floor(V / 2), axis=0)
    V_nni = upsample_geometry(V_d, 2)

    if ordering == 1:
        V_nni = V_nni[np.lexsort((V_nni[:, 2], V_nni[:, 1], V_nni[:, 0]))]
    elif ordering == 2:
        V_nni = V_nni[np.lexsort((V_nni[:, 0], V_nni[:, 2], V_nni[:, 1]))]
    elif ordering == 3:
        V_nni = V_nni[np.lexsort((V_nni[:, 1], V_nni[:, 0], V_nni[:, 2]))]  

    return V,V_nni


def get_neighbors(query_V,V,nbhd):
    """
    neighs : neighbors occupancy
    """

    temp = []
    for i in range(nbhd.shape[0]):
        temp.append(ismember_xyz((query_V + nbhd[i:i+1,:]) ,V))
    neighs = np.vstack(temp).T.reshape(-1)

    # # Faster but uses too much memory
    # neighs = ismember_xyz(
    # ( np.expand_dims(query_V,2) + np.expand_dims(nbhd.T,0) ).transpose([0,2,1]).reshape(-1,3),
    # V)

    neighs = neighs.reshape(query_V.shape[0], nbhd.shape[0])

    return neighs


def voxels_in_raster_causal_neighborhood(r):
    """
    raster order does not matter
    """
    v = 0
    for x in np.arange(-r, r+1, 1):
        for y in np.arange(-r, r+1, 1):
            for z in np.arange(-r, r+1, 1):
                if np.linalg.norm([x,y,z]) <= r:
                    if x < 0:
                        v += 1
                    elif x == 0 and y < 0:
                        v += 1
                    elif x == 0 and y == 0 and z < 0:
                        v += 1
    return v


def voxels_in_neighborhood(r):
    v = 0
    for x in np.arange(-r, r+1, 1):
        for y in np.arange(-r, r+1, 1):
            for z in np.arange(-r, r+1, 1):
                if np.linalg.norm([x,y,z]) <= r:
                    v += 1
    return v-1 # remove the center voxel


def raster_causal_nbhd(nbhd,ordering):

    assert ordering in [1,2,3]

    if ordering == 1:
        fastest = 2
        medium_speed = 1
        slowest = 0
    elif ordering == 2:
        fastest = 0
        medium_speed = 2
        slowest = 1
    elif ordering == 3:
        fastest = 1
        medium_speed = 0
        slowest = 2

    nbhd = np.delete(
        nbhd, 
        np.logical_or( 
            np.logical_or( 
                (nbhd[:, slowest] > 0) ,
                np.logical_and( 
                    (nbhd[:, slowest] == 0) , 
                    (nbhd[:, medium_speed] > 0) 
                )
            ),
            np.logical_and(
                np.logical_and( 
                    (nbhd[:, slowest] == 0) , 
                    (nbhd[:, medium_speed] == 0) 
                ),
                (nbhd[:, fastest] >= 0)
            )
        ), 
        axis=0
    )
    return nbhd


def pc_causal_context(V, N, M, ordering = 1, squeeze_nbhd: bool = False):
    """
    Find causal contexts to help guess the occupancy of each voxel in V using
    its own causal neighbors in the current level and its parent neighborhood.
    This assumes that one has already encoded the previous level of the octree
    V_d, and now they are looking for contexts to encode the children of
    V_d, i.e., for every voxel in V_d which of their 8 possible children are
    occupied.

    -Notice that V_d may be found using:
        _, idx = np.unique(np.floor(V / 2), axis=0, return_index=True)

        V_d = np.floor(V / 2)[np.sort(idx), :]


    :param V: (N-by-3) Voxelized geometry to be coded (assuming that V_d
              the previous level has already been encoded).
    :param N: How many causal neighbors from the current level to use.
    :param M: How many neighbors from the previous level to use.
    :param ordering: Options for the encoding order:
                          1 - Raster XYZ;
                          2 - Raster YZX;
                          3 - Raster ZXY;

    :return: contexts (8*L'-by-n+m) The causal nbhd neighbours found in V, plus
                                     m bits indicating the parent neighborhood. L'
                                     is the length of V_d, thus every 8 rows in
                                     contexts relates to one parent voxel in V_d.

             occupancy (8*L'-by-1)    The occupancy of each of the 8 possible children of V_d.

             this_nbhd (n-by-3)      The causal neighborhood.
    
             prev_nbhd (m-by-3)      The previous level neighbors.
    
    """

    if ordering not in [1,2,3]:
        m = f"""ordering must be 1,2 or 3"""
        raise ValueError(m)

    # neighs
    current_level_r = 1
    while voxels_in_raster_causal_neighborhood(current_level_r) < N:
        current_level_r += 1

    this_nbhd = xyz_displacements(np.arange(-current_level_r, current_level_r+1, 1))
    this_nbhd = this_nbhd[(np.linalg.norm(this_nbhd, axis=1) <= current_level_r), :]
    this_nbhd = raster_causal_nbhd(this_nbhd, ordering)
    this_nbhd = this_nbhd[np.argsort(np.linalg.norm(this_nbhd,axis=1), kind='mergesort'),:]

    V,V_nni = parents_children(V,ordering)

    occupancy = ismember_xyz(V_nni, V)

    causal_neighs = get_neighbors(V_nni,V,this_nbhd)

    causal_neighs = causal_neighs[:,:N]
    this_nbhd = this_nbhd[:N,:]

    # parent neighs (aka uncles)
    previous_level_r = 1
    while voxels_in_neighborhood(previous_level_r) < M:
        previous_level_r += 1

    prev_nbhd = xyz_displacements(np.arange(-previous_level_r, previous_level_r+1, 1))
    prev_nbhd = prev_nbhd[(np.linalg.norm(prev_nbhd, axis=1) <= previous_level_r), :]
    prev_nbhd = np.delete(prev_nbhd,prev_nbhd.tolist().index([0,0,0]), axis=0)
    prev_nbhd = prev_nbhd[np.argsort(np.linalg.norm(prev_nbhd,axis=1), kind='mergesort'),:]

    V_d, child_idx = np.unique(np.floor(V_nni / 2), axis=0, return_inverse=True)
    # child_idx holds, in the order of V_nni, indices from V_d 
    phi = get_neighbors(V_d,V_d,prev_nbhd)
    phi = phi[child_idx, :]
    
    phi = phi[:,:M]
    prev_nbhd = prev_nbhd[:M,:]

    contexts = np.column_stack((causal_neighs, phi))
    return V_nni,contexts, occupancy, this_nbhd, prev_nbhd

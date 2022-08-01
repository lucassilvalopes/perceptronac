import numpy as np
from perceptronac.coding3d import mortoncode as mc
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import warnings
import pandas as pd


def xyz_displacements(interval,ordering=1):
    x, y, z = np.meshgrid(interval, interval, interval)
    deltas = np.array([x.flatten('F'), y.flatten('F'), z.flatten('F')]).T

    if ordering == 1:
        sorting_indices = np.lexsort((deltas[:, 2], deltas[:, 1], deltas[:, 0]))
    elif ordering == 2:
        sorting_indices = np.lexsort((deltas[:, 0], deltas[:, 2], deltas[:, 1]))
    elif ordering == 3:
        sorting_indices = np.lexsort((deltas[:, 1], deltas[:, 0], deltas[:, 2]))

    return deltas[sorting_indices]


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


def parents_children(V, C, ordering):
    """
    Args:
        V : positions (or geometry) 
        C : colors (or attributes)
        ordering : raster XYZ (1), raster YZX (2) , raster ZXY (3)

    Returns:
        V : positions sorted according to input ordering
        C : colors sorted according to input ordering
        V_d : occupied voxels in the previous octree level
        V_nni : positions of the children of V_d, sorted according to input ordering
        C_nni : colors of the children of V_d, sorted according to input ordering
        occupancy : boolean mask indicating points of V_nni that are in V
    """
    if ordering == 1:
        v_sorting_indices = np.lexsort((V[:, 2], V[:, 1], V[:, 0]))
    elif ordering == 2:
        v_sorting_indices = np.lexsort((V[:, 0], V[:, 2], V[:, 1]))
    elif ordering == 3:
        v_sorting_indices = np.lexsort((V[:, 1], V[:, 0], V[:, 2]))

    V = V[v_sorting_indices]
    if C.size > 0:
        C = C[v_sorting_indices]

    V_d = np.unique(np.floor(V / 2), axis=0)
    V_nni = upsample_geometry(V_d, 2)

    if ordering == 1:
        vnni_sorting_indices = np.lexsort((V_nni[:, 2], V_nni[:, 1], V_nni[:, 0]))
    elif ordering == 2:
        vnni_sorting_indices = np.lexsort((V_nni[:, 0], V_nni[:, 2], V_nni[:, 1]))
    elif ordering == 3:
        vnni_sorting_indices = np.lexsort((V_nni[:, 1], V_nni[:, 0], V_nni[:, 2]))  

    V_nni = V_nni[vnni_sorting_indices]

    # occupancy = ismember_xyz(V_nni, V)

    # if C.size > 0:
    #     C_nni = - np.ones((V_nni.shape[0],C.shape[1])) # unoccupied voxels marked with -1

    #     C_nni[occupancy] = C
    # else:
    #     C_nni = None

    return V,C,V_nni #,C_nni,occupancy


def get_neighbors(query_V,V,C,nbhd):
    """
    neighs : neighbors occupancy
    """

    if C.size == 0:
        past_neighbor_occupancies = []
        V_hashes = mc.morton_code(V)
        for i in range(nbhd.shape[0]):
            ith_neighbor = query_V + nbhd[i:i+1,:]
            ith_neighbor_occupancy = np.isin(mc.morton_code(ith_neighbor) ,V_hashes)
            past_neighbor_occupancies.append(np.expand_dims(ith_neighbor_occupancy,1))
        all_neighs = np.concatenate(past_neighbor_occupancies,axis=1)

        # # Faster but uses too much memory
        # neighs = ismember_xyz(
        # ( np.expand_dims(query_V,2) + np.expand_dims(nbhd.T,0) ).transpose([0,2,1]).reshape(-1,3),
        # V)

        # neighs = neighs.reshape(query_V.shape[0], nbhd.shape[0])

        all_neighs_colors = np.array([[[]]])

    else:

        V_hashes = mc.morton_code(V)
        past_nbhd_occupancies = []
        past_nbhd_colors = []
        for i in range(query_V.shape[0]):
            ith_nbhd = (query_V[i:i+1,:] + nbhd)
            ith_nbhd_hashes = mc.morton_code(ith_nbhd)
            ith_nbhd_occupancy = np.isin(ith_nbhd_hashes,V_hashes)
            past_nbhd_occupancies.append(np.expand_dims(0,ith_nbhd_occupancy))

            ith_nbhd_C = - np.ones((nbhd.shape[0],C.shape[1]))

            ith_nbhd_V_mask = np.isin(V_hashes,ith_nbhd_hashes)
            distance_based_sorting=np.argsort(
                np.linalg.norm(V[ith_nbhd_V_mask]-query_V[i:i+1,:],axis=1), kind='mergesort')
            ith_nbhd_C[ith_nbhd_occupancy] = C[ith_nbhd_V_mask][distance_based_sorting,:]

            past_nbhd_colors.append(np.expand_dims(ith_nbhd_C,0))

        all_neighs = np.concatenate(past_nbhd_occupancies,axis=0)
        all_neighs_colors = np.concatenate(past_nbhd_colors,axis=0)
        
    return all_neighs,all_neighs_colors


def voxels_in_raster_neighborhood(r,include=[1,1,1,0,0,0]):
    """
    Predicts the number of voxels in a raster neighborhood.
    The raster order does not matter.

    Divides the space in 6 parts:
    - causal-half-space and non-causal-half-space, which are the half-spaces before and after the current plane being scanned
    - causal-half-plane and non-causal-half-plane, which are the half-planes before and after the current line being scanned
    - causal-half-line and non-causal-half-line, which are the half-lines before and after the current point being scanned
      (the current point being scanned is included in the non-causal-half-line).

    The vector "include" determines the absence or presence of each of these 6 parts, in the order :
    causal-half-space, causal-half-plane, causal-half-line, non-causal-half-space, non-causal-half-plane, non-causal-half-line
    """

    v = 0
    for x in np.arange(-r, r+1, 1):
        for y in np.arange(-r, r+1, 1):
            for z in np.arange(-r, r+1, 1):
                if np.linalg.norm([x,y,z]) <= r:
                    if x < 0:
                        if include[0]:
                            v += 1
                    elif x == 0 and y < 0:
                        if include[1]:
                            v += 1
                    elif x == 0 and y == 0 and z < 0:
                        if include[2]:
                            v += 1
                    elif x == 0 and y == 0 and z >= 0:
                        if include[5]:
                            v += 1
                    elif x == 0 and y > 0:
                        if include[4]:
                            v += 1
                    elif x > 0:
                        if include[3]:
                            v += 1
    return v


def raster_nbhd(nbhd,ordering,include=[1,1,1,0,0,0]):

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

    first = nbhd[:, fastest]
    second = nbhd[:, medium_speed]
    last = nbhd[:, slowest]

    causal_half_space = last < 0
    non_causal_half_space = last > 0

    causal_half_plane = np.logical_and( last == 0 , second < 0 )
    non_causal_half_plane = np.logical_and( last == 0 , second > 0 )

    causal_half_line = np.logical_and( np.logical_and( last == 0 , second == 0 ), first < 0 )
    non_causal_half_line = np.logical_and( np.logical_and( last == 0 , second == 0 ), first >= 0 )

    pieces = [
        causal_half_space,causal_half_plane,causal_half_line,
        non_causal_half_space,non_causal_half_plane,non_causal_half_line
    ]

    mask = np.zeros(nbhd.shape[0]).astype(bool)
    for i in range(len(pieces)):
        if include[i]:
            mask = np.logical_or(mask,pieces[i])

    return np.delete(nbhd, np.logical_not(mask), axis=0)



def pc_causal_context(V, N, M, ordering = 1, causal_half_space_only: bool = False, C = np.array([[]]) ):
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


    :param V: (L-by-3) Voxelized geometry to be coded (assuming that V_d
              the previous level has already been encoded).
    :param N: How many causal neighbors from the current level to use.
    :param M: How many neighbors from the previous level to use.
    :param ordering: Options for the encoding order:
                          1 - Raster XYZ;
                          2 - Raster YZX;
                          3 - Raster ZXY;
    
    :param causal_half_space_only: If True, neighbors in the plane currently being 
                                    scanned will no be used.

    :return: contexts (8*L'-by-n+m) The causal nbhd neighbours found in V, plus
                                     m bits indicating the parent neighborhood. L'
                                     is the length of V_d, thus every 8 rows in
                                     contexts relates to one parent voxel in V_d.

             occupancy (8*L'-by-1)    The occupancy of each of the 8 possible children of V_d.

             this_nbhd (n-by-3)      Causal neighbors as displacements from the central voxel.
    
             prev_nbhd (m-by-3)      Previous level neighbors also as displacements.
    
    """

    if ordering not in [1,2,3]:
        m = f"""ordering must be 1,2 or 3"""
        raise ValueError(m)

    V,C,V_nni = parents_children(V,C,ordering)
    # V,C,V_nni,C_nni,occupancy = parents_children(V,C,ordering)

    occupancy = ismember_xyz(V_nni, V)

    this_contexts_V,this_contexts_C,this_nbhd =\
        causal_siblings(V,C,V_nni,N,ordering,causal_half_space_only)

    prev_contexts_V,prev_contexts_C,prev_nbhd = uncles(V_nni,C,occupancy,M,ordering) # uncles(V_nni,C_nni,M,ordering)

    contexts = np.concatenate([this_contexts_V, prev_contexts_V],axis=1)

    contexts_colors = np.concatenate([this_contexts_C, prev_contexts_C],axis=1)

    return V_nni,contexts,contexts_colors, occupancy, this_nbhd, prev_nbhd


def determine_best_partition(N_plus_M,causal_half_space_only):

    this_include = ([1,0,0,0,0,0] if causal_half_space_only else [1,1,1,0,0,0])
    prev_include = [0,1,1,1,1,1]

    initial_N = 1
    initial_M = 1
    for prev_r in range(1,N_plus_M) : # prev_r will never reach N_plus_M. This avoids infinite loop in case of a bug.
        candidate_N = voxels_in_raster_neighborhood(2*prev_r,include=this_include)
        candidate_M = voxels_in_raster_neighborhood(prev_r,include=prev_include)
        if candidate_N + candidate_M > N_plus_M:
            break
        else:
            initial_N = candidate_N
            initial_M = candidate_M

    final_N = initial_N
    final_M = initial_M
    while final_N + final_M < N_plus_M:
        if (final_M - initial_M)/(candidate_M - initial_M) > (final_N - initial_N)/(candidate_N - initial_N) :
            final_N += 1
        else:
            final_M += 1
    return final_N, final_M


def causal_siblings(V,C,V_nni,N,ordering,causal_half_space_only=False):

    include = ([1,0,0,0,0,0] if causal_half_space_only else [1,1,1,0,0,0])

    # neighs
    current_level_r = 1
    while voxels_in_raster_neighborhood(current_level_r,include=include) < N:
        current_level_r += 1

    this_nbhd = xyz_displacements(np.arange(-current_level_r, current_level_r+1, 1),ordering=ordering)
    this_nbhd = this_nbhd[(np.linalg.norm(this_nbhd, axis=1) <= current_level_r), :]
    this_nbhd = raster_nbhd(this_nbhd,ordering,include=include)
    this_nbhd = this_nbhd[np.argsort(np.linalg.norm(this_nbhd,axis=1), kind='mergesort'),:]

    this_contexts_V,this_contexts_C = get_neighbors(V_nni,V,C,this_nbhd)

    this_contexts_V = this_contexts_V[:,:N]
    this_contexts_C = this_contexts_C[:,:N,:]
    this_nbhd = this_nbhd[:N,:]
    return this_contexts_V,this_contexts_C,this_nbhd


# def get_parents_colors(C_nni,child_idx):
#     C_table = pd.DataFrame({"values":list(C_nni) , "parent_id": child_idx})
#     C_table = C_table[np.all(C_nni!=-1,axis=1)]
#     V_d_C = C_table.groupby("parent_id")["values"].apply(lambda x: np.mean(x,axis=0)).sort_index()
#     return V_d_C


def uncles(V_nni,C,occupancy,M,ordering):
    """
    Returns the M occupancies of the M closest uncles (voxels in the previous octree level) of each point in V_nni.
    Uncles with all 8 children in the causal neighborhood are discarded.

    V_d (L'-by-3) : previous level points.
    V_nni (8*L'-by-3) : all 8 children of each point in V_d.
    prev_nbhd (M-by-3) : displacements from the central point to get the neighboring points in the previous level.
    child_idx (8*L') : for each point in V_nni, child_idx points to the parent in V_d. 
        That is V_d[child_idx[i],:] is the parent of V_nni[i,:] .
    phi (8*L'-by-M) : for each point in V_nni, phi holds the occupancy of the uncles.
        That is phi[i,:] holds the occupancy of the points in (V_d[child_idx[i],:] + prev_nbhd) .

    OBS: 'mergesort' in np.argsort and 'F' in np.reshape or np.flatten is to be compliant with matlab.
    """

    include = [0,1,1,1,1,1]
    previous_level_r = 1
    while voxels_in_raster_neighborhood(previous_level_r,include=include) < M:
        previous_level_r += 1

    prev_nbhd = xyz_displacements(np.arange(-previous_level_r, previous_level_r+1, 1),ordering=ordering)
    prev_nbhd = prev_nbhd[(np.linalg.norm(prev_nbhd, axis=1) <= previous_level_r), :]
    prev_nbhd = raster_nbhd(prev_nbhd,ordering,include=include)
    prev_nbhd = prev_nbhd[np.argsort(np.linalg.norm(prev_nbhd,axis=1), kind='mergesort'),:]

    V_d, child_idx = np.unique(np.floor(V_nni / 2), axis=0, return_inverse=True)
    # child_idx holds, in the order of V_nni, indices from V_d 
    # V_d_C = get_parents_colors(C_nni,child_idx)
    V_d_C = pd.DataFrame({"values":list(C) , "parent_id": child_idx[occupancy]}).groupby("parent_id")["values"].apply(
        lambda x: np.mean(x,axis=0)).sort_index() if (C.size > 0) else np.array([[]])
    prev_contexts_V,prev_contexts_C = get_neighbors(V_d,V_d,V_d_C,prev_nbhd)
    prev_contexts_V = prev_contexts_V[child_idx, :][:,:M]
    prev_contexts_C = prev_contexts_C[child_idx,:,:][:,:M,:]
    prev_nbhd = prev_nbhd[:M,:]
    return prev_contexts_V,prev_contexts_C,prev_nbhd

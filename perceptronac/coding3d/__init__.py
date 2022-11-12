import numpy as np
from perceptronac.coding3d import mortoncode as mc
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import warnings
import pandas as pd


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


def write_PC(filename,xyz,colors=None):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pointcloud, write_ascii=True)


def get_sorting_indices(V,ordering):
    if ordering == 1:
        v_sorting_indices = np.lexsort((V[:, 2], V[:, 1], V[:, 0]))
    elif ordering == 2:
        v_sorting_indices = np.lexsort((V[:, 0], V[:, 2], V[:, 1]))
    elif ordering == 3:
        v_sorting_indices = np.lexsort((V[:, 1], V[:, 0], V[:, 2]))
    return v_sorting_indices


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


def sort_V_C(V,C,ordering):
    v_sorting_indices = get_sorting_indices(V,ordering)

    V = V[v_sorting_indices]
    if C.size > 0:
        C = C[v_sorting_indices]
    return V,C


def interpolate_V(V,ordering):

    V_d = np.unique(np.floor(V / 2), axis=0)
    V_nni = upsample_geometry(V_d, 2)

    V_nni = V_nni[get_sorting_indices(V_nni,ordering)]
    return V_nni


def interpolate_C(C,V_nni,occupancy):
    if C.size > 0:
        C_nni = np.zeros((V_nni.shape[0],C.shape[1]),dtype=np.uint8)
        C_nni[occupancy] = C
    else:
        C_nni = np.zeros((V_nni.shape[0],0))
    return C_nni


def pc_causal_context(V, N, M, ordering = 1, causal_half_space_only: bool = False, C = None ):
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

    Args:
        V: (L-by-3) Voxelized geometry to be coded (assuming that V_d
            the previous level has already been encoded).
        N: How many causal neighbors from the current level to use.
        M: How many neighbors from the previous level to use.
        ordering: Options for the encoding order:
            1 - Raster XYZ;
            2 - Raster YZX;
            3 - Raster ZXY;
        causal_half_space_only: If True, neighbors in the plane currently being 
            scanned will no be used.

    Returns: 
        contexts (8*L'-by-n+m) : The causal nbhd neighbours found in V, plus 
            m bits indicating the parent neighborhood. L' is the length of V_d, 
            thus every 8 rows in contexts relates to one parent voxel in V_d.
        occupancy (8*L'-by-1) : The occupancy of each of the 8 possible children of V_d.
        this_nbhd (n-by-3) : Causal neighbors as displacements from the central voxel.
        prev_nbhd (m-by-3) : Previous level neighbors also as displacements.

    Other variables:
        V_d (L'-by-3) : previous level points.
        V_nni (8*L'-by-3) : all 8 children of each point in V_d.
        prev_nbhd (M-by-3) : displacements from the central point to get the 
            neighboring points in the previous level.
        child_idx (8*L') : for each point in V_nni, child_idx points to the parent in V_d. 
            That is V_d[child_idx[i],:] is the parent of V_nni[i,:] .
        phi (8*L'-by-M) : for each point in V_nni, phi holds the occupancy of the uncles.
            That is phi[i,:] holds the occupancy of the points in (V_d[child_idx[i],:] + prev_nbhd) .

    OBS: 'mergesort' in np.argsort and 'F' in np.reshape or np.flatten is to be compliant with matlab.
    
    """

    if ordering not in [1,2,3]:
        m = f"""ordering must be 1,2 or 3"""
        raise ValueError(m)

    if C is None:
        C = np.zeros((V.shape[0],0))

    V,C = sort_V_C(V,C,ordering)

    V_nni = interpolate_V(V,ordering)

    occupancy = ismember_xyz(V_nni, V)

    this_contexts_O,this_contexts_C,this_nbhd = causal_siblings(V_nni,V,C,occupancy,N,ordering,causal_half_space_only)

    prev_contexts_O,prev_contexts_C,prev_nbhd = uncles(V_nni,C,occupancy,M,ordering)

    contexts_occupancy = np.concatenate([this_contexts_O, prev_contexts_O],axis=1)

    contexts_color = np.concatenate([this_contexts_C, prev_contexts_C],axis=1)

    C_nni = interpolate_C(C,V_nni,occupancy)

    if (C.size > 0):
        return V_nni, contexts_occupancy, occupancy, this_nbhd, prev_nbhd, C_nni, contexts_color
    else:
        return V_nni, contexts_occupancy, occupancy, this_nbhd, prev_nbhd


def causal_siblings(query_V,V,C,occupancy,N,ordering,causal_half_space_only):
    include = ([1,0,0,0,0,0] if causal_half_space_only else [1,1,1,0,0,0])
    return siblings(query_V,V,C,occupancy,N,ordering,include)


def uncles(V_nni,C,occupancy,M,ordering):

    V_d, child_idx = np.unique(np.floor(V_nni / 2), axis=0, return_inverse=True) # child_idx holds, in the order of V_nni, indices from V_d 
    
    V_d_C = np.vstack(pd.DataFrame({"values":list(C) , "parent_id": child_idx[occupancy]}).groupby("parent_id")["values"].apply(
        lambda x: np.mean(x,axis=0)).sort_index().values) if (C.size > 0) else np.zeros( (V_d.shape[0],0) )

    sort_V_d = get_sorting_indices(V_d,ordering)
    prev_contexts_O,prev_contexts_C,prev_nbhd = siblings(V_d[sort_V_d],V_d[sort_V_d],V_d_C[sort_V_d],np.ones(V_d.shape[0],dtype=bool),M,ordering,[0,1,1,1,1,1])

    # I need a mapping from original V_d index to sorted V_d index
    # An array sorted as the original V_d with destinations as values
    # https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    mapping = np.argsort(sort_V_d)
    prev_contexts_O = prev_contexts_O[mapping[child_idx], :]
    prev_contexts_C = prev_contexts_C[mapping[child_idx],:,:]

    return prev_contexts_O,prev_contexts_C,prev_nbhd


def siblings(query_V,V,C,occupancy,N,ordering,include):
    
    current_level_r = 1
    while voxels_in_raster_neighborhood(current_level_r,include=include) < N:
        current_level_r += 1

    this_nbhd = xyz_displacements(np.arange(-current_level_r, current_level_r+1, 1))
    this_nbhd = this_nbhd[get_sorting_indices(this_nbhd,ordering)]
    this_nbhd = this_nbhd[(np.linalg.norm(this_nbhd, axis=1) <= current_level_r), :]
    this_nbhd = raster_nbhd(this_nbhd,ordering,include=include)
    this_nbhd = this_nbhd[np.argsort(np.linalg.norm(this_nbhd,axis=1), kind='mergesort'),:]

    if C.size == 0:
        past_neighbor_occupancies = []
        V_hashes = mc.morton_code(V)
        for i in range(this_nbhd.shape[0]):
            ith_neighbor = query_V + this_nbhd[i:i+1,:]
            ith_neighbor_occupancy = np.isin(mc.morton_code(ith_neighbor) ,V_hashes)
            past_neighbor_occupancies.append(np.expand_dims(ith_neighbor_occupancy,1))
        this_contexts_O = np.concatenate(past_neighbor_occupancies,axis=1)

        # # Faster but uses too much memory
        # neighs = ismember_xyz(
        # ( np.expand_dims(query_V,2) + np.expand_dims(nbhd.T,0) ).transpose([0,2,1]).reshape(-1,3),
        # V)

        # neighs = neighs.reshape(query_V.shape[0], nbhd.shape[0])

        this_contexts_C = np.zeros( (query_V.shape[0],this_nbhd.shape[0],0) )

    else:

        past_neighbor_occupancies = []
        V_hashes = mc.morton_code(V)
        past_neighbor_colors = []
        for i in range(this_nbhd.shape[0]):
            ith_neighbor = query_V + this_nbhd[i:i+1,:]
            ith_neighbor_hashes = mc.morton_code(ith_neighbor)
            ith_neighbor_occupancy = np.isin(ith_neighbor_hashes ,V_hashes)
            past_neighbor_occupancies.append(np.expand_dims(ith_neighbor_occupancy,1))

            ith_neighbor_C = - np.ones((query_V.shape[0],C.shape[1]))
            ith_neighbor_V_mask = np.isin(V_hashes,ith_neighbor_hashes)
            idx = np.arange(query_V.shape[0])
            direct_sort = get_sorting_indices(ith_neighbor,ordering)
            ith_neighbor_C[idx[direct_sort][ith_neighbor_occupancy[direct_sort]]] = C[ith_neighbor_V_mask]
            past_neighbor_colors.append(np.expand_dims(ith_neighbor_C,1))
        
        this_contexts_O = np.concatenate(past_neighbor_occupancies,axis=1)
        this_contexts_C = np.concatenate(past_neighbor_colors,axis=1)

        # V_hashes = mc.morton_code(V)
        # past_nbhd_occupancies = []
        # past_nbhd_colors = []
        # for i in range(query_V.shape[0]):
        #     ith_nbhd = (query_V[i:i+1,:] + this_nbhd)
        #     ith_nbhd_hashes = mc.morton_code(ith_nbhd)
        #     ith_nbhd_occupancy = np.isin(ith_nbhd_hashes,V_hashes)
        #     past_nbhd_occupancies.append(np.expand_dims(ith_nbhd_occupancy,0))

        #     ith_nbhd_C = - np.ones((this_nbhd.shape[0],C.shape[1]))

        #     ith_nbhd_V_mask = np.isin(V_hashes,ith_nbhd_hashes)
        #     distance_based_sorting=np.argsort(
        #         np.linalg.norm(V[ith_nbhd_V_mask]-query_V[i:i+1,:],axis=1), kind='mergesort')
        #     ith_nbhd_C[ith_nbhd_occupancy] = C[ith_nbhd_V_mask][distance_based_sorting,:]

        #     past_nbhd_colors.append(np.expand_dims(ith_nbhd_C,0))

        # this_contexts_O = np.concatenate(past_nbhd_occupancies,axis=0)
        # this_contexts_C = np.concatenate(past_nbhd_colors,axis=0)

    this_contexts_O = this_contexts_O[:,:N]
    this_contexts_C = this_contexts_C[:,:N,:]
    this_nbhd = this_nbhd[:N,:]
    return this_contexts_O,this_contexts_C,this_nbhd


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

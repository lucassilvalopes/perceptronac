
import numpy as np


def points2volume(points,voxelgridshape,values=None):
    ny,nx,nz = voxelgridshape
    volume = np.zeros((ny,nx,nz) if (values is None) else (ny,nx,nz,values.shape[1]),dtype=int)
    volume.reshape(
        (-1 if (values is None) else (-1,values.shape[1])),order='C')[[y*nx*nz+x*nz+z for x,y,z in points]] = \
            (1 if (values is None) else values)
    return (volume[:,:,:,0] if (volume.shape[-1] == 1) else volume)


def gpt_context(V,max_octree_level,block_side,C=None):
    """

    V (L-by-3): x,y,z of the points
    C (L-by-1): one channel colors

    https://www.quora.com/Which-is-better-Python-or-R-from-a-memory-point-of-view/answer/Jeremy-Spencer?ch=99&share=6d22b55b&srid=a7B1
    https://realpython.com/introduction-to-python-generators/
    """

    side = 2**max_octree_level

    for x in range(0,side,block_side):
        for y in range(0,side,block_side):
            for z in range(0,side,block_side):
                
                block_mask = np.logical_and(
                    np.logical_and(
                        np.logical_and(V[:,0]>=x, V[:,0]<x+block_side),
                        np.logical_and(V[:,1]>=y, V[:,1]<y+block_side)
                    ),
                    np.logical_and(V[:,2]>=z, V[:,2]<z+block_side)
                )

                if np.all(np.logical_not(block_mask)):
                    block_geo = [] 
                    block_attr = []
                    block_geo_dense = np.zeros(3*[block_side],dtype=int)
                    block_attr_dense = np.zeros(3*[block_side],dtype=int)
                else:
                    block_geo,block_attr = V[block_mask,:],C[block_mask,:]

                    block_geo[:,0] = block_geo[:,0] - x
                    block_geo[:,1] = block_geo[:,1] - y
                    block_geo[:,2] = block_geo[:,2] - z

                    block_attr_dense = points2volume(block_geo,3*[block_side],block_attr)
                    block_geo_dense = points2volume(block_geo,3*[block_side])
                
                yield block_geo, block_attr, block_geo_dense, block_attr_dense, [x,y,z]

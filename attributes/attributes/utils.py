

import numpy as np
import open3d as o3d

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
# Computes and sort the morton code from the vertices.
#
# X, Y, Z: the vertices position
# morton: the morton code
# order: how the vertices should be ordered to obtain an
#        ascending order for the morton code
# number_of_voxels: number of voxels
from numba import jit


@jit(nopython=True)
def __v2m(v: int) -> int:
    v = (v & 0x000007) | \
        (v & 0x000038) << 6 | \
        (v & 0x0001C0) << 12 | \
        (v & 0x000E00) << 18 | \
        (v & 0x007000) << 24 | \
        (v & 0x038000) << 30 | \
        (v & 0x1C0000) << 36

    v |= (v << 2) | (v << 4)
    v &= 0x9249249249249249
    return v


@jit(nopython=True)
def __m2v(m: int):
    m &= 0x9249249249249249
    m |= (m >> 2) | (m >> 4)

    m = ((m) & 0x000007) | \
        ((m >> 6) & 0x000038) | \
        ((m >> 12) & 0x0001C0) | \
        ((m >> 18) & 0x000E00) | \
        ((m >> 24) & 0x007000) | \
        ((m >> 30) & 0x038000) | \
        ((m >> 36) & 0x1C0000)
    return m


@jit(nopython=True)
def vertex2morton_code(x, y, z):
    return (__v2m(z)) | (__v2m(y) << 1) | (__v2m(x) << 2)


@jit(nopython=True)
def morton_code2vertex(morton):
    return __m2v(morton >> 2), __m2v(morton >> 1), __m2v(morton)


@jit(nopython=True)
def morton_code(V):
    morton = ([vertex2morton_code(int(x[0]), int(x[1]), int(x[2])) for x in V])
    return morton


def sort_morton(morton):
    order = sorted(range(len(morton)), key=lambda k: morton[k])
    return order

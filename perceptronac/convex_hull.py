
import math


def vnorm(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def vcos(v1,v2):
    return (v1[0]*v2[0] + v1[1]*v2[1]) / (vnorm(v1) * vnorm(v2))

def vdiff(v1,v2):
    return [v1[0]-v2[0],v1[1]-v2[1]]

def convex_hull(coord):

    # c = min(coord,key=lambda x: x[0])
    c = sorted(coord, key=lambda x: (x[0], x[1]))[0]
    ref = [0,-1] # vertical vector pointing down
    hull = [coord.index(c)]
    while ((hull[0] != hull[-1]) or (len(hull) == 1)):
        vcs=[vcos(vdiff(pt,c),ref) if pt != c else -math.inf for pt in coord]
        hull.append(vcs.index(max(vcs))) 
        p = coord[hull[-2]]
        c = coord[hull[-1]]
        ref = vdiff(c,p)
        if ref[1] >= 0: # horizontal vector pointing right
            hull.pop()
            break
    return hull

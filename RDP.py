import numpy as np


def clust_then_RDP(array):
    prev = array[0]
    for point in array:
        if np.linalg.norm(point.pos - prev.pos) > 0.1:
            prev.neighbors.append(point)
            point.neighbors.append(prev)
            prev = point
    res = DouglasPeucker(array)
    return res


def dist(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))


def DouglasPeucker(array, epsilon=0.03):
    d_max = 0
    idx_d_max = 0
    for i in range(1, len(array)-1):
        d = dist(array[i].pos, array[0].pos, array[-1].pos)
        if d > d_max:
            d_max = d
            idx_d_max = i

    if d_max > epsilon:
        res_1 = DouglasPeucker(array[:idx_d_max+1], epsilon)
        res_2 = DouglasPeucker(array[idx_d_max:], epsilon)
        res = np.concatenate([res_1[:-1], res_2])
    else:
        res = np.array([array[0], array[-1]])
    return res

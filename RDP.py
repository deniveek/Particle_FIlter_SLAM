import numpy as np


def make_clusters(array, d=0.1):
    new = [point for point in array if point[0]]
    if not len(array):
        return new
    new.insert(0,[0,0])
    array = np.array(new)
    new = []
    tmp = []
    prev = array[0]
    for point in array:
        if np.linalg.norm(point-prev) > d:
            new.append(np.array(tmp))
            tmp = []
            tmp.append(point)
        else:
            tmp.append(point)
        prev = point.copy()
    new.append(tmp)
    return new

def clust_then_RDP(array):
    tmp = make_clusters(array)
    new = []
    for cluster in tmp:
        new.append(DouglasPeucker(cluster))
    return new

def dist(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))

def DouglasPeucker(array, epsilon=0.01):
    d_max = 0
    idx_d_max = 0
    for i in range(1, len(array)-1):
        d = dist(array[i], array[0], array[-1])
        if d > d_max:
            d_max = d
            idx_d_max = i
    if d_max > epsilon:
        res_1 = DouglasPeucker(array[:idx_d_max+1], epsilon)
        res_2 = DouglasPeucker(array[idx_d_max:], epsilon)
        res = np.vstack((res_1[:-1], res_2))
    else:
        res = np.vstack([array[0], array[-1]])
    return res

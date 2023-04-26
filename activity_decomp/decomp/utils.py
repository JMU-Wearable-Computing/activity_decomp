import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from tqdm import tqdm

from activity_decomp.decomp import (DofFilter, DiffStdMetric,
               GaussianFilter, SavgolFilter,
               GaussianGlobalSegment, FFTMetric)
from itertools import chain, combinations
from sklearn_extra.cluster import KMedoids


DEFAULT_GRID = {
    "dof_filter": [DofFilter(DiffStdMetric())],
    "data_filter": [*[GaussianFilter(i) for i in range(1, 7)]],
    "deriv_func": [
    SavgolFilter(5, 2, deriv=1),
    SavgolFilter(9, 2, deriv=1),
    ],
    "global_segment": [*[GaussianGlobalSegment(i) for i in range(1, 6)]],
    "average_zvc": [True, False],
}

def default_grid_w_diff(valid=None, invalid=None):
    return {
        "dof_filter": [FFTMetric()],
        "data_filter": [*[GaussianFilter(i) for i in range(1, 7)]],
        "deriv_func": [
        SavgolFilter(5, 2, deriv=1),
        SavgolFilter(9, 2, deriv=1),
        ],
        "global_segment": [*[GaussianGlobalSegment(i) for i in range(1, 6)]],
        "average_zvc": [True, False],

        "valid": [valid],
        "invalid": [invalid]
    }


def powerset(iterable):
    if iterable is None:
        return [None]
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    l = [[b for a in l for b in a] for r in range(1, len(s)+1) for l in combinations(s, r)]
    return l


def default_grid_w_valid_powerset(valid=None, invalid=None):
    return {
        "dof_filter": [FFTMetric()],
        "data_filter": [*[GaussianFilter(i) for i in range(1, 5)]],
        "deriv_func": [
        SavgolFilter(5, 2, deriv=1),
        SavgolFilter(9, 2, deriv=1),
        ],
        "global_segment": [*[GaussianGlobalSegment(i) for i in range(1, 4)]],
        "average_zvc": [True, False],

        "valid": powerset(valid),
        "invalid": powerset(invalid) 
    }


def adaptive_cluster(min_k, max_k, points, clustering_f=KMeans, cluster_params={}):
    N = points.shape[0]
    best_k_star = 100000000
    best_classes = None
    best_k = None
    for k in range(min_k, max_k + 1):
        if N < k:
            break

        kmeans = clustering_f(n_clusters=k, **cluster_params).fit(points)
        classes = kmeans.labels_
        centriods = kmeans.cluster_centers_
        intra = 0
        inter = 0
        for c in range(k):
            mask = classes == k
            points_for_c = points[mask]
            diff = points_for_c - centriods[c]
            intra += np.sum(np.linalg.norm(diff, axis=-1) ** 2)

            diff = centriods - centriods[c]
            inter += np.sum(np.linalg.norm(diff, axis=-1) ** 2)
        
        weight = N / (k ** 2)
        k_star = intra + weight * inter
        if k_star < best_k_star or best_classes is None:
            best_k_star = k_star
            best_classes = classes
            best_k = k
    return best_classes, best_k


def get_angles_per_point(points, angles, valid, idx_to_joint):
    if isinstance(angles, np.ndarray):
        valid_idx = [idx for idx in range(angles.shape[0]) 
                        if idx_to_joint[idx] in valid]
        per_point = angles[np.ix_(valid_idx, points)].T
    elif isinstance(angles, dict):
        angles = [angles[idx_to_joint[i]]
                  for i in range(len(angles))
                  if idx_to_joint[i] in valid]
        angles = np.stack(angles, axis=0)
        per_point = angles[:, points].T
    else:
        raise Exception("angles paramter is not have type np.ndarray or dict, was {type(angles)}")
    return per_point


def order_points(points, classes, num_classes):
    classes_list = list(classes)
    order = [(c, classes_list.index(c)) for c in range(num_classes)]
    order.sort(key=lambda x: x[1])
    return [points[np.where(classes == c)[0]] for c,_ in order]


def classify_points(points, angles, seg, k, clustering_f=KMeans, cluster_params={}):
    per_point = get_angles_per_point(points, angles, seg.valid, seg.idx_to_joint)
    if per_point.shape[0] < k:
        return None, None

    classes = clustering_f(n_clusters=k, **cluster_params).fit(per_point).labels_

    return classes, order_points(points, classes, k)


def adaptive_classify_points(points, angles, seg, min_k, max_k, clustering_f=KMeans, cluster_params={}):
    per_point = get_angles_per_point(points, angles, seg.valid, seg.idx_to_joint)
    if per_point.size == 0:
        return 0, None, 0

    classes, k = adaptive_cluster(min_k, max_k, per_point,
                                  clustering_f=clustering_f,
                                  cluster_params=cluster_params)
    if classes is None or per_point.shape[0] < k:
        return 0, None, 0
    return classes, order_points(points, classes, k), k


class GridSearch():

    def __init__(self, seg, reps, k=2, search_params=DEFAULT_GRID, cluster=KMeans, cluster_params={}) -> None:
        self.seg = seg
        self.reps = reps
        self.k = k
        self.search_params = search_params
        self.param_grid = ParameterGrid(search_params)

        cluster_map = {"kmeans": KMeans, "kmedoids": KMedoids}
        cluster = cluster_map.get(cluster, cluster)
        self.cluster = cluster
        # Set a default for medoids
        if cluster == KMedoids and cluster_params == {}:
            cluster_params = {"method": "pam"}
        self.cluster_params = cluster_params
    
    def fit(self, angles, return_score=False):
        LARGE_NUM = 100000000

        best_diff = LARGE_NUM
        best_params = None
        best_seg = None
        best_points = None
        best_k = -1
        for params in tqdm(self.param_grid):
            seg = self.seg(**params)
            points = seg.fit(angles)

            if self.k is None:
                # classes, classes_lists = classify_points(points, angles, seg, k)
                classes, classes_lists, k = adaptive_classify_points(points, angles, seg, 2, 10,
                                                                     self.cluster, self.cluster_params)
            else:
                classes, classes_lists = classify_points(points, angles, seg, self.k,
                                                         self.cluster, self.cluster_params)
                k = self.k
            if classes_lists is None:
                continue

            def find_diff(i, p):
                length = len(p)
                if i == 0:
                    length -= 1
                return abs(min([abs(length / (self.reps * j)) for j in range(1, k)]) - 1)

            # starts = [i for i, c in enumerate(classes) if c==0]
            # total_consistant = 0
            # for c in starts:
            #     for i in range(k):
            #         if c + i < len(classes) and classes[c + i] == i:
            #             total_consistant += 1
            # curr_diff = abs(total_consistant / len(starts) - 1)

            curr_diff = sum([find_diff(i, p) for i, p in enumerate(classes_lists)])  
            if curr_diff < best_diff:
                best_params = params
                best_seg = seg
                best_diff = curr_diff
                best_points = classes_lists
                best_k = k
        
        self.best_seg = best_seg
        self.best_params = best_params
        self.best_diff = best_diff
        self.best_points = best_points
        if return_score:
            return best_points, best_seg, best_params, best_k, best_diff
        return best_points, best_seg, best_k, best_params
                
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from skillest.task_decomp.segmentation import * 
from tqdm import tqdm


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
        "dof_filter": [DofFilter(DiffStdMetric())],
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


def classify_points(points, angles, seg, k):
    if isinstance(angles, np.ndarray):
        valid_idx = [idx for idx in range(angles.shape[0]) 
                        if seg.idx_to_joint[idx] in seg.valid]
        per_point = angles[np.ix_(valid_idx, points)].T
    elif isinstance(angles, dict):
        angles = [angles[seg.idx_to_joint[i]]
                  for i in range(len(angles))
                  if seg.idx_to_joint[i] in seg.valid]
        angles = np.stack(angles, axis=0)
        per_point = angles[:, points].T
    else:
        raise Exception("angles paramter is not have type np.ndarray or dict, was {type(angles)}")
    if per_point.shape[0] < k:
        return None

    classes = KMeans(n_clusters=k).fit(per_point).predict(per_point)

    classes_list = list(classes)
    order = [(c, classes_list.index(c)) for c in range(k)]
    order.sort(key=lambda x: x[1])

    return [points[np.where(classes == c)[0]] for c,_ in order]


class GridSearch():

    def __init__(self, seg, reps, k=2, search_params=DEFAULT_GRID) -> None:
        self.seg = seg
        self.reps = reps
        self.k = k
        self.search_params = search_params
        self.param_grid = ParameterGrid(search_params)
    
    def fit(self, angles, return_score=False):
        LARGE_NUM = 100000000

        best_diff = LARGE_NUM
        best_params = None
        best_seg = None
        best_points = None
        for params in tqdm(self.param_grid):
            seg = self.seg(**params)
            points = seg.fit(angles)

            points = classify_points(points, angles, seg, self.k)
            if points is None:
                continue

            def find_diff(i, p):
                length = len(p)
                if i == 0:
                    length -= 1
                return abs(length - self.reps)

            curr_diff = sum([find_diff(i, p) for i, p in enumerate(points)])
            if curr_diff < best_diff:
                best_params = params
                best_seg = seg
                best_diff = curr_diff
                best_points = points
        
        self.best_seg = best_seg
        self.best_params = best_params
        self.best_diff = best_diff
        self.best_points = best_points
        if return_score:
            return best_points, best_seg, best_params, best_diff
        return best_points, best_seg, best_params
                

if __name__ == "__main__":
    import pandas as pd
    import time
    import joints as j

    # path = "/Users/rileywhite/wearable-computing/human-path-planning/data/last_recording2.csv" 
    # path = "jumping_jack_blaze.csv"
    # path = "/Users/rileywhite/wearable-computing/human-path-planning/data/jj_landmarks.csv"
    path = "blazepose_recordings/riley_squat_correct.csv"
    # path = "blazepose_recordings/riley_arm_circles_correctly.csv"
    # path = "/Users/rileywhite/wearable-computing/human-path-planning/data/jj_landmarks.csv"
    jj = pd.read_csv(path, index_col=0).values[300:]
    angles = j.blaze.get_all_angles_from_landmarks(jj, degrees=True)

    valid = ["left_shoulder_up-down", #"left_shoulder_forward-back",
            "right_shoulder_up-down", #"right_shoulder_forward-back",
            #"right_leg_left-right", "left_leg_left-right",
            "right_elbow", "left_elbow"
            # "left_knee", "right_knee",
            # "back_forward-back"
            ]

    seg = Segmentation
    grid = default_grid_w_diff()
    gs = GridSearch(seg, reps=11, k=2, search_params=grid)

    start = time.time()
    points, seg, params = gs.fit(angles)
    end = time.time()
    print(f"time {end - start}")

    # print(f"Best params: {params}")
    print(params["data_filter"].sigma)
    if isinstance(params["deriv_func"], Concat):
        print(params["deriv_func"].filters[0].sigma)
        print(params["deriv_func"].filters[1].window_length)
    else:
        print(params["deriv_func"].window_length)
    print(params["average_zvc"])

    seg.plot_segmentation(angles, True, False)
    plt.show()
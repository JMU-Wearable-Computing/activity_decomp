import numpy as np
from typing import Callable

from activity_decomp.decomp import GaussianFilter, SavgolFilter


class OnlineSeg():

    def __init__(self,
                 window=5,
                 min_legnth=1,
                 data_filter: Callable=GaussianFilter(2),
                 deriv_func: Callable=SavgolFilter(5, 2, deriv=1)) -> None:
        self.window = 5
        self.data_filter = data_filter
        self.deriv_fun = deriv_func
        self.min_length = min_legnth

        assert window > min_legnth

    def segment(self, data):
        idx_to_key = [k for k in data.keys()]
        flattened = np.array([v for v in data.values()])
        if len(flattened.shape) < 2:
            flattened = flattened.reshape(1, -1)
        filtered = self.data_filter(flattened)
        deriv = self.deriv_fun(filtered)[:, -self.window:]

        # deriv = np.array([[ 1., -1., -1., -1., -1.]])
        zvc = np.where(np.diff(np.sign(deriv), axis=-1) != 0)
        segments = []
        for row, col in zip(*zvc):
            if (self.window > col + self.min_length and
                np.abs(np.sum(np.sign(deriv[row, col+1:col+self.min_length+1]))) == self.min_length):
                segments.append(row)

        seg_set = {seg for seg in segments}
        return [idx_to_key[idx] for idx in seg_set]


if __name__ == "__main__":

    import pandas as pd
    import time

    # path = "jumping_jack_blaze.csv"
    # jj = pd.read_csv(path, index_col=0).values[:]
    # data, angle_dict, idx_dict = get_all_2d_angles(jj)

    seg = OnlineSeg(5)
    # for i in range(5, 100):
    #     print(seg.segment({k:v[:i] for k, v in angle_dict.items()}))
    input = {"test": [1, 3, 2, 1, 0]}
    seg.segment(input)

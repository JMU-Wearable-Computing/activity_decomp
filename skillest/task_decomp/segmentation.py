from typing import Callable
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
import numpy as np
from pyampd.ampd import find_peaks
from scipy.stats import gaussian_kde, linregress
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from skillest.task_decomp.blaze_pose_seg import get_all_2d_angles, segment

class Filter():

    def __call__(self, data):
        pass

class DofFilter(Filter):

    def __init__(self, *metrics) -> None:
        super().__init__()
        self.metrics = metrics

    def __call__(self, data, deriv):
        features = []
        for metric in self.metrics:
            features.append(metric(data, deriv))
        features = np.column_stack(features)
        kmeans = KMeans(n_clusters=2).fit(features)

        min_mean_args = np.argmin(kmeans.cluster_centers_, axis=0)
        min_mean_arg = np.argmax(np.bincount(min_mean_args))
        predictions = kmeans.predict(features).astype(int)

        valid_idx = np.where(predictions == min_mean_arg)[0]
        invalid_idx = np.where(predictions != min_mean_arg)[0]
        return valid_idx, invalid_idx


class DerivStdMetric():

    def __call__(self, data, deriv):
        return np.std(deriv, axis=1)


class UniformFilter(Filter):
    def __init__(self, size, axis=-1, mode="reflect", cval=0.0, origin=0) -> None:
        super().__init__()
        self.size = size
        self.axis = axis
        self.mode = mode
        self.cval = cval
        self.origin = origin

    def __call__(self, data):
        return uniform_filter1d(data,
                                size=self.size,
                                axis=self.axis,
                                mode=self.mode,
                                cval=self.cval,
                                origin=self.origin)


class SavgolFilter(Filter):
    def __init__(self,
                 window_length,
                 polyorder,
                 deriv=0,
                 delta=1.0,
                 axis=-1,
                 mode="interp",
                 cval=0.0) -> None:
        super().__init__()
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.axis = axis
        self.mode = mode
        self.cval = cval
    
    def __call__(self, data):
        return savgol_filter(data,
                             self.window_length,
                             self.polyorder,
                             self.deriv,
                             self.delta,
                             self.axis,
                             self.mode,
                             self.cval)


class SimpleDerivative(Filter):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data):
        return np.diff(data)


class Segmentation():

    def __init__(self, k,
                 covariance_factor=0.01,
                 dof_filter: DofFilter=DofFilter(DerivStdMetric()),
                 valid=None,
                 invalid=None,
                 data_filter: Callable=None,
                 deriv_func: Callable=SavgolFilter(5, 2, deriv=1),
                 scale_data=True):
        self.k = k
        self.covariance_factor = covariance_factor
        self.dof_filter = dof_filter
        self.valid = valid
        self.invalid = invalid
        self.data_filter = data_filter
        self.deriv_func = deriv_func
        self.scale_data = scale_data
    
    def fit(self, data, return_properties=False):
        if self.scale_data:
            data, self.mean, self.std = self.scale(data)
        return self.segment(data)
    
    def segment(self,
                data,
                return_properties=False):
        data = np.array(data)
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        DOF, T = data.shape
        if self.scale_data:
            data, _, _ = self.scale(data, self.mean, self.std)

        if self.data_filter is not None:
            data = self.data_filter(data)
        assert self.deriv_func is not None, "Must have a valid deriv func not None."
        deriv = self.deriv_func(data)

        # This is basically an is_fit flag, unless explicity passed in
        if self.valid is None and self.invalid is None: 
            if self.dof_filter is not None:
                self.valid, self.invalid = self.dof_filter(data, deriv)
            else:
                self.valid = np.arange(DOF)
                self.invalid = np.array([])

        filtered = np.array([data[i] for i in self.valid])
        filtered_deriv = np.array([deriv[i] for i in self.valid])

        properties = self.local_segment(filtered, filtered_deriv)
        segments, density = self.global_segment(properties["local_segments"], T)
        properties["segments"], properties["density"] = segments, density
        properties["valid"], properties["invalid"] = self.valid, self.invalid

        if return_properties:
            return segments, properties
        return segments
    
    def global_segment(self, local_segments, length):
        flattened = np.hstack(local_segments)

        density = gaussian_kde(flattened)
        xs = np.linspace(0, length, length)
        density.covariance_factor = lambda : self.covariance_factor
        density._compute_covariance()

        density = density(xs)
        density_peaks = find_peaks(density)
        return density_peaks, density
         
    def local_segment(self, pos, deriv):
        assert len(pos.shape) == 2

        all_local_segment = []
        all_points_before = []
        all_points_after = []
        all_means = []
        all_zvc = []
        properties = {"local_segments": all_local_segment, "points_before": all_points_before,
                      "points_after": all_points_after, "means": all_means, "zcvs": all_zvc}
        for p, d in zip(pos, deriv):
            means, points_before, points_after, zvc = self.find_important_sections(p, d)
            local_segment = (points_before + points_after) / 2
            all_local_segment.append(local_segment)
            all_points_before.append(points_before)
            all_points_after.append(points_after)
            all_means.append(means)
            all_zvc.append(zvc)
        
        return properties 

    def find_important_sections(self, pos, deriv):

        pos_peaks = find_peaks(deriv).reshape(-1)
        neg_peaks = find_peaks(-deriv).reshape(-1)
        peaks = np.concatenate([pos_peaks, neg_peaks])

        zvc = np.where(np.diff(np.sign(deriv)) != 0)[0]

        greater = peaks[:, np.newaxis] > zvc
        idx_before_peak = np.where(np.diff(greater) != 0)[1]
        idx_after_peak = idx_before_peak + 1

        # Remove first and last because we want mean between peaks.
        points_before = np.sort(zvc[idx_before_peak])[1:]
        points_after = np.sort(zvc[idx_after_peak])[:-1]

        means = [np.mean(pos[beg:end+1]) for beg,end in zip(points_after, points_before)]
        
        return means, points_before, points_after, zvc

    def plot_segmentation(self, data, data_labels=None):
        data = np.array(data)
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        
        fig, axes = plt.subplots(data.shape[0] + 1, 1, figsize=(8,8))

        segments, properties = self.segment(data, return_properties=True)
        valid, density = properties["valid"], properties["density"]
        local_segments, means = properties["local_segments"], properties["means"]
        if self.scale_data:
            data, _, _ = self.scale(data, self.mean, self.std)

        j = 0
        for i, (ax, key) in enumerate(zip(axes[:-1], data_labels)):
            values = data[i]

            ax.plot(values, c="g")
            if i in valid:
                ax.scatter(local_segments[j], means[j], color="g")
                j += 1
            else:
                key = f"{key} (invalid)"
            ax.title.set_text(key)
            ax.set_xticks([])
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)

        axes[-1].plot(np.arange(density.size),density)
        axes[-1].scatter(segments, density[segments], color="g")
        axes[-1].title.set_text("Gaussian Density Plot of Important Points")

        for i, ax in enumerate(axes):
            
            ymin, ymax = ax.get_ylim()
            n = 2
            skip_nth_mask = np.ones(segments.size, dtype=bool) 
            # skip_nth_mask[np.arange(0, density_peaks.size, n)] = 0
            skip_nth_mask[::n] = 0
            ax.vlines(segments[skip_nth_mask], ymin=ymin, ymax=ymax, color="y")
            ax.vlines(segments[~skip_nth_mask], ymin=ymin, ymax=ymax, color="r")
        
        return fig, axes

    def scale(self, x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=1)
            std = np.std(x, axis=1)
        return (x - mean[:, np.newaxis]) / std[:, np.newaxis], mean, std

if __name__ == "__main__":
    import pandas as pd

    jj = pd.read_csv("jumping_jack_blaze.csv", index_col="timestamp").values[25:]
    data, angle_dict, idx_dict = get_all_2d_angles(jj)

    seg = Segmentation(k=2)
    # fig, axes = plot_segmentation(data, list(angle_dict.keys()), k=2)
    seg.fit(data)
    points = seg.segment(data)
    seg.plot_segmentation(data, angle_dict.keys())
    plt.show()
    # gp = GaussianProcessSegmentModels(k=2)
    # gp.fit(points, data)

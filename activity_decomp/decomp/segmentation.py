from typing import Callable, Dict, List, Tuple, Union
from scipy.signal import savgol_filter, butter, sosfilt
from scipy.ndimage.filters import uniform_filter1d
from scipy.ndimage import gaussian_filter1d
import numpy as np
from pyampd.ampd import find_peaks
from scipy.stats import gaussian_kde
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans


class Filter():

    def __call__(self, data):
        pass

class Concat(Filter):

    def __init__(self, *filters):
        self.filters = filters
    
    def __call__(self, data):
        for filter in self.filters:
            data = filter(data)
        return data


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


class DiffStdMetric():

    def __call__(self, data, deriv):
        diff = np.diff(data, axis=1)
        return -1 * np.std(diff, axis=1) 

class DerivStdMetric():

    def __call__(self, data, deriv):
        return -1 * np.std(deriv, axis=1)

class FFTMetric(Filter):

    def __init__(self, threshold=0.9) -> None:
        self.threshold = threshold

    def __call__(self, data, deriv):
        fft = np.fft.fft(data, n=int(deriv.shape[1]), axis=-1)[:, 1:]
        power = np.abs(fft)[:, :int(deriv.shape[1] / 2)]
        primary_freq = np.argmax(np.sum(power, axis=0))
        sorted_idx = np.argsort(power[:, primary_freq])[::-1]
        # plt.plot(np.sum(power, axis=0))
        # plt.show()

        power_sum = np.sum(power[:, primary_freq])
        sorted = power[sorted_idx, :]
        m = 1
        for i in range(1, deriv.shape[0]):
            top_m = np.sum(sorted[:i, primary_freq])
            ratio = top_m / power_sum
            if ratio > self.threshold:
                break
            m = i
        return sorted_idx[:m], sorted_idx[m:]


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


class GaussianFilter(Filter):
    def __init__(self, sigma=5, axis=-1, mode="reflect", cval=0.0, origin=0) -> None:
        super().__init__()
        self.sigma = sigma
        self.axis = axis
        self.mode = mode
        self.cval = cval
        self.origin = origin

    def __call__(self, data):
        return gaussian_filter1d(data,
                                sigma=self.sigma,
                                axis=self.axis,
                                mode=self.mode,
                                cval=self.cval)


class MovingAverage(Filter):

    def __init__(self, size=10) -> None:
        super().__init__()
        self.size = size
    
    def __call__(self, data):
        def moving_average(x):
            return np.convolve(x, np.ones(self.size), 'valid') / self.size
        
        filtered = []
        for dof in data:
            filtered.append(moving_average(dof))

        return np.array(filtered)


class Butterworth(Filter):

    def __init__(self,
                 n=5,
                 wn=3,
                 btype="low",
                 fs=33) -> None:
        super().__init__()
        self.n = n
        self.wn = wn
        self.btype = btype
        self.fs = fs
    
    def __call__(self, data):

        filtered = []
        for dof in data:
            sos = butter(self.n,
                         self.wn,
                         btype=self.btype,
                         fs=30,
                         output="sos")
            lowpass = sosfilt(sos, dof)
            filtered.append(lowpass)
        return np.array(filtered)


class FitSin(Filter):
    def __init__(self, size=400) -> None:
        super().__init__()
        self.size = size
    
    def fit_sin(self, tt, yy):
        '''Fit sin to the input time sequence, and return fitting parameters
           "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c

        popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c,
                "freq": f, "period": 1./f, "fitfunc": fitfunc,
                "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    def __call__(self, data):
        T = len(data[0])
        interval = int(self.size / 2)
        x = np.linspace(0, T, T)
        filtered = []
        for dof in data:
            signal = np.zeros(T)
            # signal[:self.size] = self.fit_sin(x[:self.size], dof[:self.size])["fitfunc"](x[:self.size])
            for i in range(self.size, T, interval):
                x_slice = np.arange(self.size) 
                info = self.fit_sin(x_slice, dof[i - self.size:i])
                signal[i - self.size:i] = info["fitfunc"](x_slice)

            filtered.append(signal)
        
        return np.array(filtered)


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

class KDEGlobalSegment(Filter):
    def __init__(self, covariance_factor=0.01) -> None:
        super().__init__()
        self.covariance_factor = covariance_factor
    
    def __call__(self, data, length):
        flattened = np.hstack(data)
        density = gaussian_kde(np.tile(flattened, [1]), bw_method=0.02)
        xs = np.linspace(0, length, length)
        if self.covariance_factor is not None:
            density.covariance_factor = lambda : self.covariance_factor
            density._compute_covariance()
        density = density(xs)
        density_peaks = find_peaks(density)
        return density_peaks, density


class GaussianGlobalSegment(Filter):
    """Generates global segments from flattened local segments.

    """
    def __init__(self,
                 sigma: int=1,
                 min_density: float=0.2,
                 outlier_threshold:float=3,
                 min_important_points: int=2) -> None:
        """Generates global segments from flattened local segments.

        Args:
            sigma (int, optional): standard deviation for the gaussian filter. Defaults to 1.
            min_density (float, optional): min density required to be a peak. Defaults to 0.2.
            outlier_threshold (float, optional): Number of st.dev. of density peak
                heights below which peaks are considered outliers. Defaults to 3.
            min_important_points (int, optional): Number of local segments (important points)
                required to before finding density peaks. If the number local segments is 
                under this value an empty numpy array is returned. Defaults to 2.
        """
        super().__init__()
        self.sigma = sigma
        self.min_density = min_density
        self.outlier_threshold = outlier_threshold
        self.min_important_points = min_important_points
    
    def __call__(self, local_segments: np.ndarray, length: int) \
        -> Tuple[np.ndarray, np.ndarray]:
        """Generates global segments from local segments.

        Generate strategy:
        1. Calculate density via bincount
        2. smooth density via gaussian filter. This will output
        something simlar to the KDE graph but will not be impacted
        by the length of the sequence. This is why I perfer this method
        of density smoothing. 
        3. Remove outliers

        Args:
            data (np.ndarray): local segemnts from which global
                segments will be derived.
            length (int): Total length of original data sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: global segments and calculated density.
        """
        local_segments = np.hstack(local_segments)

        counts = np.bincount(local_segments.astype(int))
        density = np.zeros(length)
        density[:len(counts)] = counts
        density = gaussian_filter1d(density, self.sigma)

        if len(local_segments) < self.min_important_points:
            return np.array([], dtype=bool), density 

        try:
            density_peaks = find_peaks(density)
        except:
            density_peaks = np.array([], dtype=bool)

        if length - 1 in density_peaks:
            density_peaks = density_peaks[:-1]
        
        if len(density_peaks) > 0:
            mean = np.mean(density[density_peaks])
            std = np.std(density[density_peaks])
            non_outlier = density[density_peaks] > mean - std * self.outlier_threshold
            non_outlier = non_outlier & (density[density_peaks] > self.min_density)
            density_peaks = density_peaks[non_outlier]

        return density_peaks, density


class UniformGlobalSegment(Filter):
    def __init__(self, size=5) -> None:
        super().__init__()
        self.size = size
    
    def __call__(self, data, length):
        flattened = np.hstack(data)

        counts = np.bincount(flattened.astype(int))
        density = np.zeros(length)
        density[:len(counts)] = counts
        density = uniform_filter1d(density, size=self.size)

        density_peaks = find_peaks(density)
        return density_peaks, density

class IdentityGlobalFilter(Filter):

    def __call__(self, data, length):
        return np.hstack(data).astype(int), np.zeros(length)


class Segmentation():
    """Class to automatically segment data.

    The basic strategy is as follows:
    1. filter the data with data filter
    2. find the derivative from filtered data with deriv_func
    3. From the derivative find ZVC and peaks/valleys. Take the 
    average ZVC between each peak which is considered a local segment.
    4. Combine the local segments from all valid angles and convert them to
    density (bincount). Then apply a gaussian filter to smooth the density.
    This is the default step 4, but this is specified by global_segment.
    5. Finally, find the peaks of the smoothed density and remove any outliers.
    These are our global segment points.

    Example usage:
        landmarks = pd.read_csv(path).values
        angles = j.blaze.get_all_angles_from_landmarks(landmarks, degrees=True)
        seg = Segmentation() 
        segment_points = seg.fit(angles)

        fig, axes = seg.plot(angles)
        fig.show()

    """

    def __init__(self,
                 valid: List=None,
                 invalid: List=None,
                 data_filter: Filter=GaussianFilter(2),
                 deriv_func: Filter=Concat(GaussianFilter(2), SavgolFilter(5, 2, deriv=1)),
                 global_segment: Filter=GaussianGlobalSegment(2),
                 dof_filter: Filter=DofFilter(DerivStdMetric()),
                 average_zvc: bool=True,
                 scale_data=True):
        """Creates object to perform Segmentation.

        If valid/invalid angles are not specified then we will attempt to
        automatically find them via the dof_filter. If valid is given 
        invalid is the complement and vice versa. 

        Args:
            valid (List, optional): Dictionary keys for any passed data 
                that should considered for global segment. Defaults to None.
            invalid (List, optional): Dictionary keys for any passed data
                that should NOT be considered for global segment. Defaults to None.
                data_filter (Filter, optional): Filter applied to data. Defaults to GaussianFilter(2).
            deriv_func (Filter, optional): Function to calulatate the derivative.
                Defaults to Concat(GaussianFilter(2), SavgolFilter(5, 2, deriv=1)).
            global_segment (Filter, optional): Filter to generate global segments from
                local segments. Defaults to GaussianGlobalSegment(2).
            dof_filter (Filter, optional): Picks the valid/invalid DOF automatically.
                Defaults to DofFilter(DerivStdMetric()).
            average_zvc (bool,  optional): If to use average position of zvc instead
                of middle of first and last zvc in a segment. Defaults to True.
            scale_data (bool, optional): If data should be scaled. Defaults to True.
        """
        self.dof_filter = dof_filter
        self.valid = valid
        self.invalid = invalid
        self.data_filter = data_filter
        self.deriv_func = deriv_func
        self.global_segment = global_segment
        self.average_zvc = average_zvc
        self.scale_data = scale_data

        self.mean = None
        self.std = None

        self._dof = None
        self._joint_to_idx = None
        self._idx_to_joint = None
    
    def fit(self,
            data: Dict[any, Union[List, np.ndarray]],
            return_properties: bool=False) \
            -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, List[any]]]]:
        """Fits the segmentation algorithm.

        Args:
            data (Dict[any, Union[List, np.ndarray]]): Data dictionary used to fit
                the algorithm. If valid/invalid is specified by the constructor
                data must contain all given keys.
            return_properties (bool, optional): If to return extra statistics
                created when segmenting. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Dict[str, List[any]]]]: List of segment points
            and optionally an extra dictionary of properties.
        """

        self._joint_to_idx = {k: i for i, k in enumerate(data.keys())}
        self._idx_to_joint = {i: k for i, k in enumerate(data.keys())}

        converted = np.array([row for row in data.values()])
        if self.scale_data:
            _, self.mean, self.std = self.scale(converted)

        return self.segment(data, return_properties)
    
    def set_valid_joints(self, unmodified_data: np.ndarray, deriv: np.ndarray) -> None:
        """Sets the valid/invalid point global variables.

        Args:
            unmodified_data (np.ndarray): Unfiltered data passed to segment.
            deriv (np.ndarray): Derivative of data.
        """
        # This is basically an is_fit flag, unless explicity passed in
        if self.valid is None and self.invalid is None: 
            if self.dof_filter is not None:
                self.valid, self.invalid = self.dof_filter(unmodified_data, deriv)
            else:
                self.valid = np.arange(self.dof)
                self.invalid = np.array([])
            self.valid = [self.idx_to_joint[i] for i in self.valid]
            self.invalid = [self.idx_to_joint[i] for i in self.invalid]
        elif self.valid is None:
            self.valid = [key for key in self.joint_to_idx.keys()
                          if key not in self.invalid]
        elif self.invalid is None:
            self.invalid = [key for key in self.joint_to_idx.keys()
                            if key not in self.valid]

    def preprocess(self, data: Dict[any, Union[List, np.ndarray]]) \
        -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data.

        Applys the filter, calulcates the derivative, and sets
        valid/invalid DOF. Derivative is calulated from the filtered
        position data.

        Args:
            data (Dict[any, Union[List, np.ndarray]]): The data passed to
                segment.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The filtered position data
                and the derivative.
        """
        data = np.array([data[self.idx_to_joint[i]] for i in range(self.dof)])

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        unmodified_data = data
        if self.scale_data:
            data, _, _ = self.scale(data, self.mean, self.std)

        if self.data_filter is not None:
            data = self.data_filter(data)
        assert self.deriv_func is not None, "Must have a valid deriv func not None."
        deriv = self.deriv_func(data)

        self.set_valid_joints(data, deriv)

        return data, deriv
    
    def segment(self,
                data: Dict[any, Union[List, np.ndarray]],
                return_properties: bool=False) \
                -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, List[any]]]]:
        """Segments the data assuming `fit` was called.

        Args:
            data (Dict[any, Union[List, np.ndarray]]): Data dictionary for segmentaion.
                If valid/invalid is specified by the constructor data must contain all 
                given keys.
            return_properties (bool, optional): If to return extra statistics
                created when segmenting. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Dict[str, List[any]]]]: List of segment points
            and optionally an extra dictionary of properties.
        """
        filtered, filtered_deriv = self.preprocess(data)
        properties = self.local_segment(filtered, filtered_deriv)

        valid_local_segments = [seg for idx, seg in enumerate(properties["local_segments"])
                                if self.idx_to_joint[idx] in self.valid]
        segments, density = self.global_segment(valid_local_segments, filtered.shape[1])
        properties["segments"], properties["density"] = segments, density
        properties["valid"], properties["invalid"] = self.valid, self.invalid

        if return_properties:
            return segments, properties
        return segments
    
    def local_segment(self, pos: np.ndarray, deriv: np.ndarray) \
        -> Dict[str, List[any]]:
        """Takes all data and calls local_segment_helper in a loop.

        Args:
            pos (np.ndarray): The positional data. Expects 2 dims.
            deriv (np.ndarray): The derivative of the positional data.
                Expects 2 dims.

        Returns:
            Dict[str, List[any]]: Dict of results generated by
            local_segment_helper.
        """
        assert len(pos.shape) == 2
        assert len(deriv.shape) == 2

        props = {"local_segments": [], "points_before": [], "points_after": [],
                 "means": [], "zvcs": [], "peaks": []}
        for p, d in zip(pos, deriv):
            local_segment, means, points_before, points_after, zvc, peaks = self.local_segment_helper(p, d)
            props["local_segments"].append(local_segment)
            props["points_before"].append(points_before)
            props["points_after"].append(points_after)
            props["means"].append(means)
            props["zvcs"].append(zvc)
            props["peaks"].append(peaks)

        return props 

    def local_segment_helper(self, pos: np.ndarray, deriv: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Segments a single DOF.

        General strategy:
            1. Find peaks and valleys of derivative
            2. Find zvc
            3. Select zvcs right before and after the peaks
            4. Deal with edge cases
            5. Find middle of zvcs between peaks.

        Args:
            pos (np.ndarray): The positional data. Expects 1 dim.
            deriv (np.ndarray): The derivative of the positional data.
                Expects 1 dim.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Returns the 6 results needed for downstream calulcations and plotting.
            local_segment, mean of angle, points_before, points_after, zvc, peaks.
        """

        pos_peaks = find_peaks(deriv).reshape(-1)
        neg_peaks = find_peaks(-deriv).reshape(-1)
        peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
        zvc = np.nonzero(np.diff(np.sign(deriv)))[0]

        # greater is a matrix where each ith row is which peak x positions
        # are greater than the ith zvc. Repeat rows are removed. Repeats
        # happen if there is not at least one zvc between peaks.
        # idx_before_peak is the peak before the transition from being less
        # than to greater than the ith zvc. Aka the first index before a peak. 
        greater = np.unique(peaks[:, np.newaxis] > zvc, axis=0)
        idx_before_peak = np.nonzero(np.diff(greater))[1]
        idx_after_peak = idx_before_peak + 1

        # If there are no zvcs before first peak then we must use 
        # the first zcv as the start of the first important segment.
        # To do this we add it to the idx_before_peak
        if np.all(greater[0, :] == 0):
            tmp = np.zeros(idx_after_peak.size + 1, dtype=int)
            tmp[1:] = idx_after_peak
            idx_after_peak = tmp
        else:
            idx_before_peak = idx_before_peak[1:]

        if np.all(greater[-1, :] == 1):
            tmp = np.zeros(idx_before_peak.size + 1, dtype=int)
            tmp[:-1] = idx_before_peak
            tmp[-1] = zvc.size - 1
            idx_before_peak = tmp
        else:
            idx_after_peak = idx_after_peak[:-1]

        points_before = zvc[idx_before_peak]
        points_after = zvc[idx_after_peak]

        if self.average_zvc:
            # Average position between zvc
            between = (zvc[:, np.newaxis] >= points_after) & (zvc[:, np.newaxis] <= points_before)
            local_segments = np.zeros(points_after.size)
            for i, mask in enumerate(between.T):
                local_segments[i] = np.mean(zvc[mask])
        else:
            local_segments = (points_before + points_after) / 2

        means = np.array([np.mean(pos[beg:end+1]) for beg,end in zip(points_after, points_before)])
        
        return local_segments, means, points_before, points_after, zvc, peaks
    
    def plot_extra(self,
                   ax: plt.Axes,
                   idx: int,
                   deriv: np.ndarray,
                   properties: Dict[str, List[any]]) -> None:
        """Plots extra info about segmentation.

        Args:
            ax (plt.Axes): Axes to plot on.
            idx (int): DOF currently being plotted.
            deriv (np.ndarray): Derivative of data.
            properties (Dict[str, List[any]]): Properties of segmentation.
        """
        peaks = properties["peaks"]
        zvcs = properties["zvcs"]

        ax2 = ax.twinx()
        ax2.plot(deriv[idx], c="r")

        ax2.scatter(zvcs[idx], deriv[idx][zvcs[idx]], color="b")
        ax2.scatter(peaks[idx], deriv[idx][peaks[idx]], color="r")

    def plot_segmentation(self,
                          data: Dict[any, Union[List, np.ndarray]],
                          plot_extra: bool=False,
                          show_filtered_data: bool=True)\
                       -> Tuple[matplotlib.figure.Figure, plt.Axes]:
        """Plots the segmentation of the given data.
        Returns the fig and axes of the plot. Call plt.show()
        to see the graph.

        Args:
            data (Dict[any, Union[List, np.ndarray]]): Data dictionary for segmentaion.
                If valid/invalid is specified by the constructor or fit data must contain all 
                given keys.
            plot_extra (bool, optional): If to plot extra statistics. Defaults to False.
            show_filtered_data (bool, optional): If to show the filtered or raw data.
              Defaults to True.

        Returns:
            Tuple[matplotlib.figure.Figure, plt.Axes]: The figure and axes of the graph.
        """

        segments, properties = self.segment(data, return_properties=True)
        invalid, means = properties["invalid"], properties["means"]
        local_segments, density = properties["local_segments"], properties["density"]

        data = np.array([data[self.idx_to_joint[i]] for i in range(self.dof)])
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        filtered_data = data
        if self.data_filter is not None:
            filtered_data = self.data_filter(data)
        deriv = self.deriv_func(filtered_data)
        
        fig, axes = plt.subplots(data.shape[0] + 1, 1, figsize=(8,8))

        for idx, ax in enumerate(axes[:-1]):
            key = self.idx_to_joint[idx]
            if show_filtered_data:
                values = filtered_data[idx]
            else:
                values = data[idx]
            ax.plot(values, c="g")

            mean = 0
            if self.scale_data:
                mean = (means[idx] * self.std[idx]) + self.mean[idx]
            ax.scatter(local_segments[idx], mean, color="g")

            if plot_extra:
                self.plot_extra(ax, idx, deriv, properties)
            if key in invalid:
                key = f"{key} (invalid)"

            ax.title.set_text(key)
            ax.set_xticks([])
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)

        axes[-1].plot(np.arange(density.size),density)
        axes[-1].scatter(segments, density[segments], color="g")
        axes[-1].title.set_text("Gaussian Density Plot of Important Points")

        for idx, ax in enumerate(axes):
            
            ymin, ymax = ax.get_ylim()
            n = 2
            skip_nth_mask = np.ones(segments.size, dtype=bool) 
            skip_nth_mask[::n] = 0
            ax.vlines(segments[skip_nth_mask], ymin=ymin, ymax=ymax, color="y")
            ax.vlines(segments[~skip_nth_mask], ymin=ymin, ymax=ymax, color="r")
        
        return fig, axes

    def scale(self, x: np.ndarray, mean: np.ndarray=None, std: np.ndarray=None)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scales data by optional mean/std.

        Args:
            x (np.ndarray): Data to scale
            mean (np.ndarray, optional): mean to scale data by. If None it will
                be calulcated from x. Defaults to None.
            std (np.ndarray, optional): std to scale data by. If None it will be
                calulated from x. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: scaled data, mean, std
        """
        if mean is None and std is None:
            mean = np.mean(x, axis=1)
            std = np.std(x, axis=1)
        return (x - mean[:, np.newaxis]) / std[:, np.newaxis], mean, std

    @property
    def idx_to_joint(self) -> Dict[int, any]:
        """Gives the idx to joint map from fitted data.
        Not defined before fit is called. 

        Raises:
            Exception: Error if fit has not been called.

        Returns:
            Dict[int, any]: Map from idx to joint
        """
        if self._idx_to_joint is None:
            raise Exception("Run fit before trying to get idx_to_joint, it has not been set yet")
        return self._idx_to_joint

    @property
    def joint_to_idx(self) -> Dict[any, int]:
        """Gives the joint to idx map from fitted data.
        Not defined before fit is called. 

        Raises:
            Exception: Error if fit has not been called.

        Returns:
            Dict[int, any]: Map from joint to idx
        """
        if self._joint_to_idx is None:
            raise Exception("Run fit before trying to get joint_to_idx, it has not been set yet")
        return self._joint_to_idx

    @property
    def dof(self) -> int:
        """The number of degrees of freedom of the data.
        E.g. the number of keys expected in the data dict.

        Raises:
            Exception: Error if fit has not been called.

        Returns:
            int: DOF.
        """
        if self._dof is None:
            if hasattr(self, "idx_to_joint"):
                self._dof = len(self.idx_to_joint.keys())
            else:
                raise Exception("Run fit before trying to get DOF, it has not been set yet")
        return self._dof


class SegmentationMinValue(Segmentation):

    def local_segment(self, pos, deriv):
        assert len(pos.shape) == 2

        all_local_segment = []
        all_means = []
        all_peaks = []
        properties = {"local_segments": all_local_segment, "means": all_means, 
                      "peaks": all_peaks}
        for p, d in zip(pos, deriv):
            local_segment, means, peaks = self.local_segment_helper(p, d)
            all_local_segment.append(local_segment)
            all_means.append(means)
            all_peaks.append(peaks)
        
        return properties 

    def local_segment_helper(self, pos, deriv):
        LARGE_NUM = 10000000
        pos_peaks = find_peaks(deriv).reshape(-1)
        neg_peaks = find_peaks(-deriv).reshape(-1)
        peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))

        edge_cases = [peaks]
        if peaks[0] != 0:
            edge_cases.insert(0, [0])
        if peaks[-1] != pos.size - 1:
            edge_cases.append([pos.size - 1])
        peaks = np.concatenate(edge_cases)

        largest_seg = np.max(np.diff(peaks))
        segment_slices = np.ones([peaks.size - 1, largest_seg], dtype=int) * -1
        precaluated_range = np.arange(largest_seg)
        for i in range(peaks.size - 1):
            peak_0 = peaks[i]
            peak_1 = peaks[i + 1]
            length = peak_1 - peak_0
            segment_slices[i, :length] = precaluated_range[:length] + peak_0

        deriv = np.concatenate([deriv, [LARGE_NUM]])
        segment_slice_min_idx = np.argmin(np.abs(uniform_filter1d(deriv[segment_slices], size=5, mode="constant", cval=LARGE_NUM)), axis=1)
        segment_min_idx = segment_slices[np.arange(len(segment_slices)), segment_slice_min_idx]

        return segment_min_idx, pos[segment_min_idx], peaks
    
    def plot_extra(self, ax, i, valid_idx, data, properties):
        valid = properties["valid"]
        if self.data_filter is not None:
            data = self.data_filter(data)
        deriv = self.deriv_func(data)
        peaks = properties["peaks"]

        ax2 = ax.twinx()
        ax2.plot(deriv[i], c="r")
        if i in valid:
            ax2.scatter(peaks[valid_idx], deriv[i][peaks[valid_idx]], color="r")


class SegmentationMinLength(Segmentation):

    def __init__(self, k,
                dof_filter: DofFilter = DofFilter(DerivStdMetric()), valid=None, invalid=None,
                data_filter: Callable = GaussianFilter(5),
                deriv_func: Callable = Concat(GaussianFilter(1), SavgolFilter(5, 2, deriv=1)),
                global_segment_filter: Filter=GaussianGlobalSegment(3),
                scale_data=True,
                min_length=5):
        super().__init__(k, dof_filter, valid, invalid, data_filter, deriv_func, global_segment_filter, scale_data)
        self.min_length = min_length

    def local_segment(self, pos, deriv):
        assert len(pos.shape) == 2

        all_local_segment = []
        all_means = []
        properties = {"local_segments": all_local_segment, "means": all_means}
        for p, d in zip(pos, deriv):
            local_segment, means = self.local_segment_helper(p, d)
            all_local_segment.append(local_segment)
            all_means.append(means)
        
        return properties 

    def local_segment_helper(self, pos, deriv):

        zvc = np.where(np.diff(np.sign(deriv)) != 0)[0]
        segments = []
        for c in zvc:
            if (len(deriv) > c + self.min_length and
                np.abs(np.sum(np.sign(deriv[c+1:c+self.min_length+1]))) == self.min_length):
                segments.append(c)

        return np.array(segments), pos[segments]
    
    def plot_extra(self, ax, i, data, properties):

        if self.data_filter is not None:
            data = self.data_filter(data)
        deriv = self.deriv_func(data)
        ax2 = ax.twinx()
        ax2.plot(deriv[i], c="r")



if __name__ == "__main__":
    import pandas as pd
    import time
    import joints as j

    # path = "/Users/rileywhite/wearable-computing/human-path-planning/data/last_recording2.csv" 
    path = "jumping_jack_blaze.csv"
    # path = "/Users/rileywhite/wearable-computing/human-path-planning/data/jj_landmarks.csv"
    # path = "blazepose_recordings/riley_squat_correct.csv"
    # path = "blazepose_recordings/riley_arm_circles_correctly.csv"
    jj = pd.read_csv(path, index_col=0).values[300:]
    angles = j.blaze.get_all_angles_from_landmarks(jj, degrees=True)

    valid = ["left_shoulder_up-down", #"left_shoulder_forward-back",
            "right_shoulder_up-down", #"right_shoulder_forward-back",
            # "right_leg_left-right", "left_leg_left-right",
            "right_elbow", "left_elbow"
            # "left_knee", "right_knee",
            # "back_forward-back"
            ]
    # data, angle_dict, idx_dict = get_all_2d_angles(jj)

    # seg = SegmentationMinLengthOneAxis(k=2,
    #                                    data_filter=GaussianFilter(2),
    #                                    valid=list(range(6)),
    #                                    dof=2,
    #                                    global_segment_filter=IdentityGlobalFilter())
    # seg = SegmentationMinLength(k=2, data_filter=GaussianFilter(3), valid=list(range(6)), min_length=1)
    # seg = SegmentationMinValue(k=2, data_filter=GaussianFilter(10), valid=list(range(6)))
    deriv_filter = Concat(GaussianFilter(4), SavgolFilter(5, 2, deriv=1))
    seg = Segmentation( valid=valid)
    # fig, axes = plot_segmentation(data, list(angle_dict.keys()), k=2)
    # seg.fit(data)
    start = time.time()
    points, _ = seg.fit(angles, return_properties=True)
    print(points.shape)
    end = time.time()
    print(end - start)
    seg.plot_segmentation(angles, True)
    # for i in range(5, len(data[0]), 5):
    #     seg.plot_segmentation(np.tile(data[:, :i], [1, 1]), angle_dict.keys(), True)
    #     # time.sleep(0.1)
    #     plt.pause(0.0001)
    #     plt.close()
    plt.show()
    # gp = GaussianProcessSegmentModels(k=2)
    # gp.fit(points, data)

import difflib
from pprint import pp
from turtle import pos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# from scipy.signal import find_peaks, peak_widths
import scipy.signal as sig
from pyampd.ampd import find_peaks
from scipy.stats import gaussian_kde, linregress
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


all_landmarks = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", 
                "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
                "right_ear", "mouth_left", "mouth_right", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                "right_wrist", "left_pinky", "right_pinky", "left_index",
                "right_index", "left_thumb", "right_thumb", "left_hip", 
                "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle", "left_heel", "right_heel", "left_foot_index",
                "right_foot_index"]


def get_idx(landmarks, z=True, vis=False, flatten=True):
    if type(landmarks) == str:
        landmarks = [landmarks]
    
    idxs = []
    for landmark in landmarks:
        idx = all_landmarks.index(landmark) * 4
        idx = [idx, idx + 1]
        if z:
            idx.append(idx + 2)
        if vis:
            idx.append(idx + 3)
        idxs.append(idx)

    idxs = np.array(idxs)
    if flatten:
        idxs = idxs.flatten()
    return idxs


def gen_angle(d, relative=False):
    x,y,z = d

    if not relative:
        x = x - y
        z = z - y
    cosine_angle = np.divide(np.sum(x * z, axis=-1), 
                            (np.linalg.norm(x, axis=-1) * np.linalg.norm(z, axis=-1)),
                            where=np.sum(x * z, axis=-1) != 0)
    angle = np.arccos(cosine_angle)
    return np.rad2deg(angle)

def get_angle(landmarks: list, data):
    idxs = get_idx(landmarks, z=False, flatten=False)
    return gen_angle(data[:, idxs].transpose([1,0,2]))

def get_all_2d_angles(data):
    left_waist = get_angle(["left_shoulder", "left_hip", "left_knee"], data)
    right_waist = get_angle(["right_shoulder", "right_hip", "right_knee"], data)

    left_shoulder = get_angle(["left_elbow", "left_shoulder", "left_hip"], data)
    right_shoulder = get_angle(["right_elbow", "right_shoulder", "right_hip"], data)

    left_elbow = get_angle(["left_shoulder", "left_elbow", "left_wrist"], data)
    right_elbo = get_angle(["right_shoulder", "right_elbow", "right_wrist"], data)

    left_knee = get_angle(["left_hip", "left_knee", "left_ankle"], data)
    right_knee = get_angle(["right_hip", "right_knee", "right_ankle"], data)

    left_ankle = get_angle(["left_knee", "left_ankle", "left_heel"], data)
    right_ankle = get_angle(["right_knee", "right_ankle", "right_heel"], data)

    idxs = get_idx(["left_shoulder", "left_hip"], z=False, flatten=False)
    pos = data[:, idxs].transpose([1,0,2])
    left_waist_unit_vector = np.expand_dims(pos[1] + np.array([-1, 0]), axis=0)
    left_torso = gen_angle(np.concatenate([pos, left_waist_unit_vector], axis=0))

    idxs = get_idx(["right_shoulder", "right_hip"], z=False, flatten=False)
    pos = data[:, idxs].transpose([1,0,2])
    right_waist_unit_vector = np.expand_dims(pos[1] + np.array([1, 0]), axis=0)
    right_torso = gen_angle(np.concatenate([pos, right_waist_unit_vector], axis=0))

    angle_dict = {"left_waist": left_waist, "right_waist": right_waist, 
                  "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
                  "left_elbow": left_elbow, "right_elbow": right_elbo,
                  "left_knee": left_knee, "right_knee": right_knee,
                  "left_ankle": left_ankle, "right_ankle": right_ankle,
                  "left_torso": left_torso, "right_torso": right_torso}
    idx_dict = {k: i for i, k in enumerate(angle_dict.keys())}
    angles = np.stack(list(angle_dict.values()), axis=0)

    return angles, angle_dict, idx_dict


def arg_reject_outliers(data, m=3.5):
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / (mdev if mdev else 1.)
    # return np.where(0.67454 * s < m)
    return np.where(s < m)


def important_point_mean_diff(data, **kwargs):
    k = kwargs["k"]
    local_points = kwargs["local_points"]

    point_diff = np.zeros([data.shape[0]])
    for i, points in enumerate(local_points):
        # points = np.array(points[:points.shape[0] - points.shape[0] % 2], dtype=int)
        points = np.array(points, dtype=int)
        mean = 0
        for j in range(k):
            mean += np.mean(np.abs(data[i, points][::k][:-1] - data[i, points][k::k]))
        point_diff[i] = mean
    return point_diff

def segment(data, k,
            covariance_factor=0.01,
            filter_funcs: list=None, 
            func_kwargs: list[dict]=None,
            valid=None, invalid=None,
            return_density=False,
            return_local_info=False):

    all_local_points, all_points_before, all_points_after, all_means = select_local_points(data)

    if valid is None:
        if filter_funcs is None:
            assert filter_funcs is None and func_kwargs is None
            filter_funcs = [important_point_mean_diff]
            func_kwargs = [{"k": k, "local_points": all_local_points}]
        valid, invalid = filter(data, filter_funcs=filter_funcs, func_kwargs=func_kwargs)

    valid_points = [all_local_points[i] for i in valid]
    density_peaks, density = select_global_points(valid_points, data.shape[1], covariance_factor)

    return_list = [density_peaks, valid, invalid]
    if return_density:
        return_list.append(density)
    if return_local_info:
        return_list.append(all_local_points)
        return_list.append(all_points_before)
        return_list.append(all_points_after)
        return_list.append(all_means)
    return return_list

def select_global_points(local_points, length, covariance_factor=0.01):
    flattened = np.hstack(local_points)

    density = gaussian_kde(flattened)
    xs = np.linspace(0, length, length)
    density.covariance_factor = lambda : covariance_factor
    density._compute_covariance()

    density = density(xs)
    density_peaks = find_peaks(density)
    return density_peaks, density

def select_local_points( data):
    assert len(data.shape) == 2

    all_local_points = []
    all_points_before = []
    all_points_after = []
    all_means = []
    for ts in data:
        means, points_before, points_after = find_important_sections(ts)
        local_points = (points_before + points_after) / 2
        all_local_points.append(local_points)
        all_points_before.append(points_before)
        all_points_after.append(points_after)
        all_means.append(means)
    
    return all_local_points, all_points_before, all_points_after, all_means
        
def find_important_sections(angles):

    angle_diff = savgol_filter(angles, 5, 2, mode="mirror", axis=0, deriv=1)
    pos_peaks = find_peaks(angle_diff).reshape(-1)
    neg_peaks = find_peaks(-angle_diff).reshape(-1)
    peaks = np.concatenate([pos_peaks, neg_peaks])

    diff_zero = np.where(np.diff(np.sign(angle_diff)) != 0)[0]

    greater = peaks[:, np.newaxis] > diff_zero
    idx_before_peak = np.where(np.diff(greater) != 0)[1]
    idx_after_peak = idx_before_peak + 1

    # Remove first and last because we want mean between peaks.
    points_before = np.sort(diff_zero[idx_before_peak])[1:]
    points_after = np.sort(diff_zero[idx_after_peak])[:-1]

    means = [np.mean(angles[beg:end+1]) for beg,end in zip(points_after, points_before)]
    
    return means, points_before, points_after

def filter(data, filter_funcs: list, func_kwargs: list[dict]):

    features = []
    for func, kwargs in zip(filter_funcs, func_kwargs):
        features.append(func(data, **kwargs))
    features = np.column_stack(features)
    gmm = GaussianMixture(n_components=2, n_init=10).fit(features)

    min_mean_args = np.argmin(np.abs(gmm.means_), axis=0)
    min_mean_arg = np.argmax(np.bincount(min_mean_args))
    predictions = gmm.predict(features).astype(int)

    valid_idx = np.where(predictions == min_mean_arg)[0]
    invalid_idx = np.where(predictions != min_mean_arg)[0]
    return valid_idx, invalid_idx

def plot_segmentation(data, data_labels, k,
                      filter_funcs: list=None, 
                      func_kwargs: list[dict]=None,
                      valid=None, invalid=None,
                      covariance_factor=0.01):

    fig, axes = plt.subplots(data.shape[0] + 1, 1, figsize=(8,8))

    ret_list = segment(data, k=k,
                       covariance_factor=covariance_factor,
                       valid=valid, invalid=invalid,
                       filter_funcs=filter_funcs,
                       func_kwargs=func_kwargs,
                       return_density=True,
                       return_local_info=True)
    density_peaks, valid, invalid, density, local_points, points_before, points_after, means = ret_list

    for i, (ax, key) in enumerate(zip(axes[:-1], data_labels)):
        values = data[i]

        ax.plot(values, c="g")
        ax.scatter(local_points[i], means[i], color="g")
        if i in invalid:
            key = f"{key} (invalid)"
        ax.title.set_text(key)
        ax.set_xticks([])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)

    axes[-1].plot(np.arange(density.size),density)
    axes[-1].scatter(density_peaks, density[density_peaks], color="g")
    axes[-1].title.set_text("Gaussian Density Plot of Important Points")

    for i, ax in enumerate(axes):
        
        ymin, ymax = ax.get_ylim()
        n = 2
        skip_nth_mask = np.ones(density_peaks.size, dtype=bool) 
        # skip_nth_mask[np.arange(0, density_peaks.size, n)] = 0
        skip_nth_mask[::n] = 0
        ax.vlines(density_peaks[skip_nth_mask], ymin=ymin, ymax=ymax, color="y")
        ax.vlines(density_peaks[~skip_nth_mask], ymin=ymin, ymax=ymax, color="r")
    
    return fig, axes

class SegmentLowMovement():

    def __init__(self, k, 
                 covariance_factor=0.01,
                 filter_funcs: list=None, 
                 func_kwargs: list[dict]=None):
        self.k = k
        self.covariance_factor = covariance_factor
        self.filter_funcs = filter_funcs
        self.func_kwargs = func_kwargs
    
    def fit(self, data):
        self.fitted_peaks, self.valid, self.invalid = segment(data,
                                                              self.k,
                                                              covariance_factor=self.covariance_factor,
                                                              filter_funcs=self.filter_funcs, 
                                                              func_kwargs=self.func_kwargs)
        return self.fitted_peaks
    
    def segment_new_data(self, data):
        peaks, _, _ = segment(data, self.k, 
                              covariance_factor=self.covariance_factor,
                              valid=self.valid, 
                              invalid=self.invalid)
        return peaks
    
    def plot(self, data, data_dict):
        return plot_segmentation(data, data_dict,
                                 k=self.k,
                                 valid=self.valid,
                                 invalid=self.invalid,
                                 covariance_factor=self.covariance_factor)


class GaussianProcessSegmentModels():

    def __init__(self, k, speed_sensitive=False) -> None:
        self.k = k
        self.speed_sensitive = speed_sensitive

    def combine_segments(self, points, data):
        segments_x = [[] for i in range(self.k)]
        segments_y = [[] for i in range(self.k)]
        mean_lengths = np.zeros(self.k)

        for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
            j = i
            total_segments = 0
            while j+1 < points.size:
                beg = points[j]
                end = points[j+1]
                seg_y.extend(list(data[beg:end]))
                seg_x.extend(range(end - beg))
                mean_lengths[i] += end - beg

                j += self.k
                total_segments += 1
            if total_segments == 0:
                print(f"{i}, {j}: ")
                print(f"{points[j]}:{points[j+1]}")
            mean_lengths[i] /= total_segments

        return segments_x, segments_y, mean_lengths

    def fit(self, points, data):
        self.models = []
        self.mean_lengths = []
        for d in data:
            model, mean = self.fit_one(points, d)
            self.models.append(model)
            self.mean_lengths.append(mean)

    def fit_one(self, points, data):
        segments_x, segments_y, mean_lengths = self.combine_segments(points, data)

        models = []
        # kernel = DotProduct() + WhiteKernel()
        # Multiplying by constant works way better because it seems to disable optmization
        # which is failing to converge
        kernel = RBF(1.0) +  WhiteKernel(1.0)
        for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
            seg_x = np.array(seg_x)
            seg_y = np.array(seg_y)
            gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
            gp.fit(seg_x.reshape(-1, 1), seg_y.reshape(-1, 1))
            models.append(gp)
        return models, mean_lengths
        
    def predict(self, points):
        means = []
        sigmas = []
        for i in range(len(self.models)):
            mean, sigma = self.predict_one(points, self.models[i], self.mean_lengths[i], return_std=True)
            means.append(mean)
            sigmas.append(sigma)
        return means, sigmas

    def predict_one(self, points, models, mean_lengths, return_std=False, return_cov=False):
        assert self.models is not None, "Fit has not been called"

        preds = []
        sigmas = []
        for peak in range(0, len(points) - 1, self.k):
            for i in range(self.k):
                beg = points[peak + i]
                end = points[peak + i + 1]
                length =  end - beg if self.speed_sensitive else mean_lengths[i]

                x = np.linspace(0, length, end - beg).reshape(-1, 1)
                pred, sigma = models[i].predict(x, return_std=return_std, return_cov=return_cov)

                preds.extend(list(pred))
                sigmas.extend(list(sigma))

        return_list = [np.array(preds), np.array(sigmas)]

        return return_list
    
    def score(self, points, data):
        scores = []
        for i, d in enumerate(data):
            score = self.score_one(points, d, self.models[i], self.mean_lengths[i])
            scores.append(score)
        return np.array(scores)
    
    def score_one(self, points, data, models, mean_lengths):
        assert self.models is not None, "Fit has not been called"

        scores = []
        for peak in range(0, len(points) - 1, self.k):
            for i in range(self.k):
                beg = points[peak + i]
                end = points[peak + i + 1]
                length =  beg - end if self.speed_sensitive else mean_lengths[i]

                x = np.linspace(0, length, end - beg).reshape(-1, 1)
                score = models[i].score(x, data[beg:end])

                scores.append(score)

        return scores

    def log_proba(self, points, data):
        log_probs = []
        for i, d in enumerate(data):
            logprobs = self.log_proba_one(points, d, self.models[i], self.mean_lengths[i])
            log_probs.append(logprobs)
        return log_probs
    
    def log_proba_one(self, points, data, models, mean_lengths):
        assert self.models is not None, "Fit has not been called"

        logprobs = []
        for peak in range(0, len(points) - 1, self.k):
            for i in range(self.k):
                beg = points[peak + i]
                end = points[peak + i + 1]
                length =  end - beg if self.speed_sensitive else mean_lengths[i]

                x = np.linspace(0, length, end - beg).reshape(-1, 1)
                mean, cov = models[i].predict(x, return_cov=True)
                mean = mean.reshape([-1])

                print(f"mean: {mean.reshape([-1])}")
                print(f"data: {data[beg:end]}")
                rv = multivariate_normal(mean=mean.reshape([-1]), cov=np.diag(np.diag(cov)))
                logprob = rv.logpdf(data[beg:end])
                print(f"prob: {logprob}")
                print(f"mse: {np.sqrt(np.sum((mean - data[beg:end])**2))}")

                print(f"shape: {np.abs(data[beg:end] - mean).shape}")
                print(f"{np.sqrt(np.diag(cov)).shape}")
                over_std = np.sum(np.abs(data[beg:end] - mean) > 1.96 * np.sqrt(np.diag(cov)))
                print(f"over_std: {over_std}")
                logprobs.append(over_std)

        return logprobs


def scale(x, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
    return (x - mean[:, np.newaxis]) / std[:, np.newaxis], mean, std

# def find_global_important_sections(local_points, length, covariance_factor=0.01):

#     density = gaussian_kde(local_points)
#     xs = np.linspace(0, length, length)
#     density.covariance_factor = lambda : covariance_factor
#     density._compute_covariance()

#     density = density(xs)
#     density_peaks = find_peaks(density)
#     return density_peaks, density


# def segment(angles, covariance_factor=0.01, return_density=False, return_local_info=False):
#     assert len(angles.shape) == 2
#     B, T = angles.shape
#     all_local_points = []
#     all_points_before = []
#     all_points_after = []
#     all_means = []
#     for ts in angles:
#         means, points_before, points_after = find_important_sections(ts)
#         local_points = (points_before + points_after) / 2
#         all_local_points.append(local_points)
#         all_points_before.append(points_before)
#         all_points_after.append(points_after)
#         all_means.append(means)
    
#     flattened = np.hstack(all_local_points)
#     density_peaks, density = find_global_important_sections(flattened, T, covariance_factor)

#     return_list = [density_peaks]
#     if return_density:
#         return_list.append(density)
#     if return_local_info:
#         return_list.append(all_local_points)
#         return_list.append(all_points_before)
#         return_list.append(all_points_after)
#         return_list.append(all_means)
#     return return_list

if __name__ == "__main__":
    # jj = pd.read_csv("jumping_jack_blaze.csv", index_col="timestamp").values[25:]
    jj = pd.read_csv("jumping_jack_blaze.csv", index_col="timestamp").values[25:]
    data, angle_dict, idx_dict = get_all_2d_angles(jj)
    data, mean, std = scale(data)

    seg = SegmentLowMovement(k=2)
    # fig, axes = plot_segmentation(data, list(angle_dict.keys()), k=2)
    seg.fit(data)
    points = seg.segment_new_data(data)
    # seg.plot(data, angle_dict.keys())
    gp = GaussianProcessSegmentModels(k=2)
    gp.fit(points, data)


    jj = pd.read_csv("blazepose_recordings/correct_jumping_jack.csv", index_col="timestamp").values[25:]
    data, angle_dict, idx_dict = get_all_2d_angles(jj)
    data, mean, std = scale(data, mean, std)

    fig, axes = plot_segmentation(data, list(angle_dict.keys()), k=2)
    points = seg.segment_new_data(data)

    scores = gp.score(points, data)
    per_rep = np.mean(scores[seg.valid], axis=0)
    per_joint = np.mean(scores, axis=1)
    print(f"Per rep: {per_rep}")
    print(f"Per joint: {per_joint}")
    for ax, score in zip(axes[:-1], scores):
        ax.set_xticks((points[:-1] + points[1:]) / 2)
        ax.set_xticklabels([f"{s:.2}" for s in score])


    # ret_list = segment(angles, k=2, return_density=True, return_local_info=True, covariance_factor=0.01)
    # density_peaks, valid_idx, invalid_idx, density, local_points, points_before, points_after, means = ret_list


    #####
    # pos_sizes = np.array([pos.size for pos in local_points])
    # size_diff = np.expand_dims(pos_sizes - density_peaks.size, axis=-1)
    # print(size_diff)

    # idx_diff = [np.abs(pos[:, np.newaxis] - density_peaks) for pos in local_points]
    # closest_diff = [np.argmin(diff, axis=1) for diff in idx_diff]
    # mean_diff = np.array([np.mean(diff[np.arange(diff.shape[0]), closest]) for diff, closest in zip(idx_diff, closest_diff)])
    # print(mean_diff)
    # k = 2
    # point_diff = np.zeros([angles.shape[0]])
    # for i, points in enumerate(local_points):
    #     # points = np.array(points[:points.shape[0] - points.shape[0] % 2], dtype=int)
    #     points = np.array(points, dtype=int)
    #     mean = 0
    #     for j in range(k):
    #         mean += np.mean(np.abs(angles[i, points][::k][:-1] - angles[i, points][k::k]))
    #     point_diff[i] = mean

    # # diff = np.std(np.diff(angles, axis=1), axis=1)
    # # _, s, _ = np.linalg.svd(angles[..., np.newaxis])
    # features = np.column_stack([point_diff])
    # print(f"features: {features}")
    # # non_outliers = arg_reject_outliers(pos_sizes - density_peaks.size, m=1)
    # # print(non_outliers)
    # gmm = GaussianMixture(n_components=2, n_init=10).fit(features)
    # print(f"means: {gmm.means_}")
    # min_mean_args = np.argmin(np.abs(gmm.means_), axis=0)
    # min_mean_arg = np.argmax(np.bincount(min_mean_args))
    # print(f"min_arg: {min_mean_arg}")
    # predictions = gmm.predict(features)
    # print(predictions)

    # valid_angles = np.where(predictions == min_mean_arg)[0]
    # invalid_angles = np.where(predictions != min_mean_arg)[0]

######
    # all_models = []

    # fig, axes = plt.subplots(data.shape[0] + 1, 1, figsize=(8,8))

    pred, sigma = gp.predict(points)
    for i, ax in enumerate(axes[:-1]):
        x = points[0] + np.arange(len(pred[i]))
        ax.plot(x, np.array(pred[i]))
        print(np.mean(sigma[i]))
        ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([pred[i] - 1.9600 * sigma[i],
                        (pred[i] + 1.9600 * sigma[i])[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    
    # down_sampled = angles[0, density_peaks[0]:density_peaks[1]]
    # x = np.arange(0, len(down_sampled))
    # print(all_models[0][0].score(x.reshape(-1, 1), down_sampled.reshape(-1, 1)))

    # down_sampled = angles[0, density_peaks[0]:density_peaks[1]][::2]
    # x = np.arange(0, len(down_sampled)) * 2
    # print(x)
    # print(down_sampled.shape)
    # print(all_models[0][0].score(x.reshape(-1, 1), down_sampled.reshape(-1, 1)))

    # sample = angles[0, density_peaks[0]:density_peaks[1]]
    # x = np.linspace(0, len(sample), len(sample) * 2)
    # y = np.interp(x, np.arange(len(sample)), sample)
    # print(x)
    # print(x.shape)
    # print(y.shape)
    # print(all_models[0][0].score(x.reshape(-1, 1), y.reshape(-1, 1)))

    plt.show()

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

def find_top_peaks(x, height=1, top_k=100):
    if x.ndim == 1:
        x = x.reshape(1, -1, 1)
    if x.ndim != 3:
        raise Exception("Ambiguous dims")

    B,T,C = x.shape
    print(x.shape)
    best_peaks = np.ones([B, C, top_k]) * -1
    for b in range(B):
        for c in range(C):
            ppeaks, _ = sig.find_peaks(x[b, :, c], height=height, prominence=.5)
            npeaks, _ = sig.find_peaks(-x[b, :, c], height=height, prominence=.5)

            peaks = np.concatenate([ppeaks, npeaks])
            top = top_k if peaks.size >= top_k else peaks.size
            curr_best_peaks = np.argpartition(-np.abs(x[b, peaks, c]), top - 1)[:top]
            print(curr_best_peaks)
            best_peaks[b, c, :top] = peaks[curr_best_peaks]
    print(best_peaks)
    return best_peaks[best_peaks > -1]


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def arg_reject_outliers(data, m=3.5):
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / (mdev if mdev else 1.)
    # return np.where(0.67454 * s < m)
    return np.where(s < m)

def segment(angles):

    angle_diff = savgol_filter(angles, 5, 2, mode="mirror", axis=0, deriv=1)
    pos_peaks = find_peaks(angle_diff).reshape(-1)
    neg_peaks = find_peaks(-angle_diff).reshape(-1)
    peaks = np.concatenate([pos_peaks, neg_peaks])

    diff_zero = np.where(np.diff(np.sign(angle_diff)) != 0)[0]
    idx_diff = peaks[:, np.newaxis] - diff_zero

    greater = peaks[:, np.newaxis] > diff_zero
    idx_before_peak = np.where(np.diff(greater) != 0)[1]
    idx_after_peak = idx_before_peak + 1

    # Remove first and last because we want mean between peaks.
    points_before = np.sort(diff_zero[idx_before_peak])[1:]
    points_after = np.sort(diff_zero[idx_after_peak])[:-1]
    closest_points = np.concatenate([points_before, points_after])
    means = [np.mean(angles[beg:end+1]) for beg,end in zip(points_after, points_before)]
    
    return means, points_before, points_after

def combine_segments(points, data, k=2):
    segments_x = [[] for i in range(k)]
    segments_y = [[] for i in range(k)]
    for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
        j = i
        while j+1 < points.size:
            beg = points[j]
            end = points[j+1]
            seg_y.extend(list(data[beg:end]))
            seg_x.extend(range(end - beg))

            j += k
    return segments_x, segments_y

def fit_line(points, data, k=2):
    segments_x, segments_y = combine_segments(points, data, k=2)

    models = []
    for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
        lr = np.polyfit(seg_x, seg_y, 3)
        models.append(lr)
    
    beg = points[0]
    end = points[1]
    r1 = np.arange(end - beg)
    # plt.plot(r1, np.polyval(models[0], r1))
    beg = points[1]
    end = points[2]
    r2 = np.arange(end - beg)
    # plt.plot(r2, np.polyval(models[1], r2))
    # plt.show()
    return [r1, r2], [np.polyval(models[0], r1), np.polyval(models[1], r2)]

def fit_gp(points, data, k=2):
    segments_x, segments_y = combine_segments(points, data, k=2)

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
    
    preds = []
    sigmas = []
    for peak in range(0, len(points) - 1 - k, k):
        for i in range(k):
            beg = points[peak + i]
            end = points[peak + i + 1]
            r1 = np.arange(end - beg).reshape(-1, 1)
            pred, sigma = models[i].predict(r1, return_std=True)
            preds.extend(list(pred))
            sigmas.extend(list(sigma))

    return np.array(preds), np.array(sigmas)


def scale(x):
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    return (x - mean[:, np.newaxis]) / std[:, np.newaxis]

if __name__ == "__main__":
    jj = pd.read_csv("jumping_jack_blaze.csv", index_col="timestamp").values[25:]
    angles, angle_dict, idx_dict = get_all_2d_angles(jj)
    angles = scale(angles)

    fig, axes = plt.subplots(angles.shape[0] + 1, 1, figsize=(8,8))
    all_positions = []
    for i, (ax, key) in enumerate(zip(axes[:-1], angle_dict.keys())):
        values = angles[i]
        means, points_before, points_after = segment(values)
        positions = (points_after + points_before) / 2
        all_positions.append(positions)

        ax.plot(values, c="g")
        ax.scatter(positions, means, color="g")
        ax.title.set_text(key)
        ax.set_xticks([])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)

    positions = np.hstack(all_positions)
    density = gaussian_kde(positions)
    xs = np.linspace(0, angles.shape[1], angles.shape[1])
    density.covariance_factor = lambda : .01
    density._compute_covariance()

    density = density(xs)
    axes[-1].plot(xs,density)

    density_peaks = find_peaks(density)
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

    pos_sizes = np.array([pos.size for pos in all_positions])
    size_diff = np.expand_dims(pos_sizes - density_peaks.size, axis=-1)
    print(size_diff)

    idx_diff = [np.abs(pos[:, np.newaxis] - density_peaks) for pos in all_positions]
    closest_diff = [np.argmin(diff, axis=1) for diff in idx_diff]
    mean_diff = np.array([np.mean(diff[np.arange(diff.shape[0]), closest]) for diff, closest in zip(idx_diff, closest_diff)])
    print(mean_diff)

    features = np.column_stack([size_diff, mean_diff])
    # non_outliers = arg_reject_outliers(pos_sizes - density_peaks.size, m=1)
    # print(non_outliers)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, n_init=10).fit(features)
    min_mean_args = np.argmin(np.abs(gmm.means_), axis=0)
    min_mean_arg = np.argmax(np.bincount(min_mean_args))
    predictions = gmm.predict(features)

    valid_angles = np.where(predictions == min_mean_arg)[0]
    invalid_angles = np.where(predictions != min_mean_arg)[0]
    for i in invalid_angles:
        axes[i].title.set_text(axes[i].get_title() + " (invalid)")

    for i, ax in enumerate(axes[:-1]):
        pred, sigma = fit_gp(density_peaks, angles[i])
        x = density_peaks[0] + np.arange(len(pred))
        ax.plot(x, np.array(pred))
        print(np.mean(sigma))
        ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([pred - 1.9600 * sigma,
                        (pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

    plt.show()

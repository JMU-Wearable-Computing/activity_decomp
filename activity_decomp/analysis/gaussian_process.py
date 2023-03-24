import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from scipy.stats import multivariate_normal


class GaussianProcess():

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


if __name__ == "__main__":
    from activity_decomp.decomp import Segmentation
    import joints as j


    jj = pd.read_csv("jumping_jack_blaze.csv", index_col="timestamp").values[25:]
    angle_dict = j.blaze.get_all_angles_from_landmarks(jj)
    data = np.array([row for k, row in angle_dict.itmes()])
    seg = Segmentation()
    points = seg.fit(angle_dict)

    gp = GaussianProcess(k=2)
    gp.fit(points, data)

    jj = pd.read_csv("blazepose_recordings/correct_jumping_jack.csv", index_col="timestamp").values[25:]
    angle_dict = j.blaze.get_all_angles_from_landmarks(jj)
    data = np.array([row for k, row in angle_dict.itmes()])

    fig, axes = seg.plot_segmentation(angle_dict)
    points = seg.segment(angle_dict)

    scores = gp.score(points, data)
    per_rep = np.mean(scores[seg.valid], axis=0)
    per_joint = np.mean(scores, axis=1)
    print(f"Per rep: {per_rep}")
    print(f"Per joint: {per_joint}")
    for ax, score in zip(axes[:-1], scores):
        ax.set_xticks((points[:-1] + points[1:]) / 2)
        ax.set_xticklabels([f"{s:.2}" for s in score])

######

    pred, sigma = gp.predict(points)
    for i, ax in enumerate(axes[:-1]):
        x = points[0] + np.arange(len(pred[i]))
        ax.plot(x, np.array(pred[i]))
        print(np.mean(sigma[i]))
        ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([pred[i] - 1.9600 * sigma[i],
                        (pred[i] + 1.9600 * sigma[i])[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    
    plt.show()

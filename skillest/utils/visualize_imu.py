import matplotlib.pyplot as plt
import numpy as np



def plot_activity(activity_data: np.array, timestep_lim: int = 100, plot: bool = True, fig=None, ax=None):


    if fig is None and ax is None:
        fig, ax = plt.subplots()
    ax.set_title("Activity Plot")
    ax.set_ylabel("Accerleration")
    ax.set_xlabel("Timestep")

    timesteps = np.arange(timestep_lim)
    for i in range(activity_data.shape[1]):
        ax.plot(timesteps, activity_data[:timestep_lim, i])

    if fig is not None:
        fig.legend(["x acceleration", "y acceleration", "z acceleration"])
    if plot:
        plt.show()

    return ax


if __name__ == "__main__":
    from os.path import join
    from skillest.utils import get_activity_data_info
    import pandas as pd
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('activity')
    parser.add_argument('user', type=int)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('dir')
    args = parser.parse_args()

    data_filepath, feature_cols = get_activity_data_info(args.dir)
    data = pd.read_csv(data_filepath)
    data = data[data["Activity"] == args.activity]
    data = data[data["UserID"] == args.user]

    data = data[feature_cols].values
    plot_activity(data, args.n)

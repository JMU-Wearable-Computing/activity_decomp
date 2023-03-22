from typing import Dict, List, Tuple, Union
from transitions import Machine
import numpy as np


from skillest.analysis.distance import EuclideanDistance
from skillest.fsm.model import Model, generate_transitions
from skillest.fsm.rules import StaticRule

from skillest.task_decomp.segmentation import Segmentation
from skillest.task_decomp.utils import GridSearch, default_grid_w_diff, classify_points
import joints as j


class Decomposer():
    """Decomposes a given recorded activity.
    The decomposer segments the data, and then each segment point is classified
    into one of k key poses via a GMM. Then each bundle of segments is averaged
    into k final key points which are passed off to the FSM modeling. 

    Optionally the decomposer can grid search over different segmentation
    strategies to find one that best lines up with k key poses and
    the given number of reps.

    Example usage for 10 reps and 2 expected poses:
        import cv2

        landmarks = pd.read_csv(path).values
        angles = j.blaze.get_all_angles_from_landmarks(landmarks, degrees=True)

        decomposer = Decomposer(k=2, grid_search=True, reps=10)
        activity, fsm = decomposer.decompose(angles, landmarks)

        activity, m = get_decomposed()
        for k in range(2):
            j.blaze.vizualize(activity.rules[f"pose_{k}"].landmarks, name=f"pose_{k}")
        
        cv2.waitKey(0)
    """

    def __init__(self, k: int,
                 seg: Segmentation=None,
                 valid_angles: List[any]=None,
                 grid_search: bool=True,
                 reps: int=None) -> None:
        """Makes a decomposer object.

        Args:
            k (int): Number of expected different key poses.
            seg (Segmentation, optional): Segmentation object. If it is None
                then a default one will be created. Cannot be non-none if grid_seaerch
                is true or non-none. Defaults to None.
            valid_angles (List[any], optional): Angles you want to whitelist
                for the segmentation. Defaults to None.
            grid_search (bool, optional): If to grid search or not. Optionally
                you may give a GridSearch object. Cannot be True or a custom GridSearch 
                object if seg is not None. Defaults to True.
            reps (int, optional): Number of expected reps for grid search. 
                Ignored if a GridSearch object is given. Defaults to None.

        Raises:
            Exception: If grid search is true but reps is not specified a GridSearch
                object cannot be created.
            Exception: If seg is non-none when grid_search is True or a custom Object.
        """
        self.k = k
        self.grid_search = None
        self.seg = None
        if seg is None and grid_search == True:
            seg = Segmentation
            if reps is None:
                raise Exception("""grid_search is set to true which requires the number
                                   of target reps. Please specify `reps` in the parameters.""")
            self.grid_search = GridSearch(seg, reps, k, default_grid_w_diff(valid_angles))
        elif seg is None and grid_search == False:
            self.seg = Segmentation(valid=valid_angles)
        elif seg is None and grid_search is not None:
            self.grid_search = grid_search
        elif seg is not None and grid_search == False:
            self.seg: Segmentation = seg
        else:
            raise Exception("seg paramter cannot be non-none when grid_search is True or a custom GridSearch object.")
    
    def decompose(self,
                  angle_dict: Dict[any, Union[List, np.ndarray]],
                  landmarks: np.array=None) -> Tuple[Model, Machine]:
        """Decomposes poses from the given data.

        Args:
            angle_dict (Dict[any, Union[List, np.ndarray]]): Angles to decompose with.
            landmarks (np.array, optional): Landmarks to decompose. If landmarks
            are given then the average landmark for each key pose will be calculated.
            This is useful for creating plots of the poses. Defaults to None.

        Returns:
            Tuple[Model, Machine]: The model and complement FSM.
        """

        if self.grid_search is None:
            points = self.seg.fit(angle_dict)
            points = classify_points(points, angle_dict, self.seg, self.k)
        else:
            points, self.seg, _ = self.grid_search.fit(angle_dict)

        angle_avg, angle_std, landmarks_avg = [0,] * self.k, [0,] * self.k, [0,] * self.k,
        for i, p in enumerate(points):
            angle_avg[i], angle_std[i] = self.avg_pose(angle_dict, p)
            if landmarks is not None:
                landmarks_avg[i] = j.blaze.average_landmarks(landmarks[p, :])

        poses = self.make_poses(angle_avg, angle_std, self.seg.valid)

        if landmarks is not None:
            for pose, avg in zip(poses[1:], landmarks_avg):
                pose.set_landmarks(avg)

        transitions, states = generate_transitions(poses)

        activity = Model({name: state for name, state in zip([state.name for state in states], states)})
        m = Machine(model=activity, states=states, transitions=transitions, initial="init", send_event=True)

        return activity, m

    def avg_pose(self,
                 angle_dict: Dict[any, Union[List, np.ndarray]],
                 points: np.ndarray) \
                    -> Tuple[Dict[any, float], Dict[any, Tuple[float, float]]]:
        """Finds the average pose for the given points.

        Args:
            angle_dict (Dict[any, Union[List, np.ndarray]]): Angle data to average over.
            points (np.ndarray): Key points to average over.

        Returns:
            Tuple[Dict[any, float], Dict[any, Tuple[float, float]]]: The averaged angles,
                angle stdev, and an optional averaged landmarks.
        """
        ordered_angles = [angle_dict[self.seg.idx_to_joint[i]]
                          for i in range(len(angle_dict))]
        angles = np.stack(ordered_angles, axis=0)

        joint_to_idx = self.seg.joint_to_idx
        key_angles = angles[:, points]
        means = np.mean(key_angles, axis=1)
        diff = key_angles - means[:, None]

        pos_std = {} 
        neg_std = {} 
        for joint, idx in self.seg.joint_to_idx.items():
            # pos_std[joint] = np.mean(diff[idx, diff[idx] >= 0]) * 3
            # neg_std[joint] = np.mean(diff[idx, diff[idx] < 0]) * 3
            pos_std[joint] = 20
            neg_std[joint] = -20

        angle_avg = {k: means[i] for k, i in joint_to_idx.items()}
        angle_std = {k: (neg_std[k], pos_std[k]) for k in joint_to_idx.keys()}
            
        return angle_avg, angle_std

    def make_poses(self, angles: List[Dict[any, float]],
                   angle_dev: List[Dict[any, Tuple[float, float]]],
                   valid: List[any]) -> Tuple[Model, Machine]:
        """Constructs a model and FSM from angle averages.

        Args:
            angles (List[Dict[any, float]]): List of angle dicts with mean
            angle_dev (List[Dict[any, Tuple[float, float]]]): List of angle dicts
            with pos/neg stdev.
            valid (List[any]): Valid angles to use.

        Returns:
            Tuple[Model, Machine]: Model and FSM
        """

        initial_key_pose = {angle: angles[0][angle] for angle in valid}
        initial_pose_deviation = {angle: angle_dev[0][angle] for angle in valid}
        initial_pose = StaticRule(f"init", key_data=initial_key_pose,
                        distance=EuclideanDistance(),
                        key_deviation=initial_pose_deviation)
        print(f"initial key pose: {initial_key_pose}")

        poses = [initial_pose]
        for i, (key, dev) in enumerate(zip(angles, angle_dev)):
            key_pose = {angle: key[angle] for angle in valid}
            pose_deviation = {angle: dev[angle] for angle in valid}
            pose = StaticRule(f"pose_{i}", key_data=key_pose,
                            distance=EuclideanDistance(),
                            key_deviation=pose_deviation)

            print(f"key pose {i}: {key_pose}")

            pose.set_parent(poses[-1])
            poses[-1].set_child(pose)
            poses.append(pose)

        poses[1].set_parent(poses[-1])
        poses[-1].set_child(poses[1])
        return poses

def get_decomposed():

    import pandas as pd
    import matplotlib.pyplot as plt
    # path = "/Users/rileywhite/wearable-computing/skill-estimation/jumping_jack_blaze.csv"
    wpath = "/Users/rileywhite/wearable-computing/human-path-planning/data/jj_wlandmarks.csv"
    path = "/Users/rileywhite/wearable-computing/human-path-planning/data/jj_landmarks.csv"

    # path = "jumping_jack_blaze.csv"
    # wpath = "jumping_jack_blaze.csv"

    jj = pd.read_csv(path).values[20:]
    wjj = pd.read_csv(wpath).values[20:]
    # angles, angles_dict, idx_dict = get_all_2d_angles(wjj) 

    angles_dict = j.blaze.get_all_angles_from_landmarks(jj,degrees=True)

    valid = ["left_shoulder_up-down", # "left_shoulder_forward-back",
            "right_shoulder_up-down",# "right_shoulder_forward-back",
            # "right_leg_left-right", "left_leg_left-right",
            "right_elbow", "left_elbow"]
    decomposer = Decomposer(k=2, valid_angles=None, grid_search=True, reps=11)
    activity, m = decomposer.decompose(angles_dict, jj)

    fig, axes = decomposer.seg.plot_segmentation(angles_dict, True)
    fig.show()

    return activity, m

if __name__ == "__main__":
    import cv2

    ### Uncomment this if you want a live recording decomposed
    # landmarks = []
    # frames = 0
    # for landmark, wlandmark, image in j.blaze.capture(return_image=True):
    #     frames += 1
    #     if frames < 50:
    #         continue
    #     cv2.putText(image, "Go", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, 255, thickness=10)
    #     cv2.imshow("blaze", image)

    #     landmarks.append(landmark)
    # landmarks = np.array(landmarks)
    # angles_dict = j.blaze.get_all_angles_from_landmarks(landmarks, degrees=True)

    # decomposer = Decomposer(k=2, valid_angles=None, grid_search=True, reps=10)
    # activity, m = decomposer.decompose(angles_dict, landmarks)

    # fig, axes = decomposer.seg.plot_segmentation(angles_dict, True)
    # fig.show()

    ### Uncomment this if you want a prerecorded activity
    activity, m = get_decomposed()

    cv2.destroyAllWindows()
    for k in range(2):
        j.blaze.vizualize(activity.rules[f"pose_{k}"].landmarks, name=f"pose_{k}")
    
    cv2.waitKey(0)


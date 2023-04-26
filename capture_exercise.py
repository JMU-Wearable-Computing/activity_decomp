import numpy as np
import cv2
import os
import argparse

import joints as j
from activity_decomp.decomp import Decomposer

K = {"jumping_jack": 2, "bicep_curl": 2, "high_knees": 3, "6_step_lunge": 4, "arm_raise": 5}

parser = argparse.ArgumentParser()
parser.add_argument("subject", type=int, help="Subject number tha twill be recorded")
parser.add_argument("exercise", type=str,
                    help="one of [jumping_jack, bicep_curl, high_knees, 6_step_lunge, arm_raise]",
                    choices=list(K.keys()))
parser.add_argument("-o", "--overwrite", default=False, action='store_true',
                    help="If to overwrite the recording matching the subject and exercise")
args = parser.parse_args()

DIR = "data/"
OVERWRITE = args.overwrite

def record(subject, exercise):
    file_name = f"{DIR}{subject}_{exercise}"
    if not OVERWRITE and os.path.exists(file_name + ".avi"):
        raise Exception("Subject/exercise already exists, delete first or specifiy overwrite if you want to overwrite")
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    window_name = "prep"
    image = np.zeros((400, 400))
    cv2.putText(image, "Get Ready", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, 255, thickness=10) 
    cv2.imshow(window_name, image)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    # Hack to make sure the window closes quickly
    # Gives opencv2 more time to destroy window
    for i in range (4):
        cv2.waitKey(1)

    k = K[exercise]
    video = j.blaze.capture_video(video_file_name=file_name)
    landmarks = []
    for landmark, wlandmark in j.blaze.capture(stream=video, file_name=file_name, return_image=False):
        landmarks.append(landmark)

    landmarks = np.array(landmarks)
    angles_dict = j.blaze.get_all_angles_from_landmarks(landmarks, degrees=True)

    decomposer = Decomposer(k=k, valid_angles=list(angles_dict.keys()), grid_search=True, reps=10)
    activity, m = decomposer.decompose(angles_dict, landmarks)

    cv2.destroyAllWindows()
    for i in range(k):
        j.blaze.vizualize(activity.rules[f"pose_{i}"].landmarks, name=f"pose_{i}")

    cv2.waitKey(0)

    decomposer = Decomposer(k=None, valid_angles=list(angles_dict.keys()), grid_search=True, reps=10)
    activity, m = decomposer.decompose(angles_dict, landmarks)

    cv2.destroyAllWindows()
    for i in range(k):
        j.blaze.vizualize(activity.rules[f"pose_{i}"].landmarks, name=f"pose_{i}")

    cv2.waitKey(0)

record(args.subject, args.exercise)
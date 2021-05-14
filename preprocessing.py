from pathlib import Path
from sys import argv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# %% Define helper functions


def load_image(
    img_path: str,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    **kwargs,
) -> mp.framework.formats.landmark_pb2.NormalizedLandmarkList:
    """All preprocessing task to load a single frame.

    Args:
        img_path (str): Path of the image.
    """
    # load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract pose
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        **kwargs,
    ) as pose:
        # image_height, image_width, _ = image.shape
        results = pose.process(image)

        return results.pose_landmarks


def get_landmark(
    name: str, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
):
    """Get landmark postition from English name."""
    landmark_names = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky_1",
        "right_pinky_1",
        "left_index_1",
        "right_index_1",
        "left_thumb_2",
        "right_thumb_2",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]
    try:
        landmark_i = landmarks.landmark[landmark_names.index(name)]
        return np.array([landmark_i.x, landmark_i.y, landmark_i.z])
    except:
        return [None, None, None]


def normalize_body(landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList):
    """Orient the body in the same way, and normalize distances."""
    # Scale
    ## calculate torso size
    # shoulder_center = (
    #     get_landmark("left_shoulder", landmarks)
    #     - get_landmark("right_shoulder", landmarks)
    # ) / 2.0
    # hip_center = (
    #     get_landmark("left_hip", landmarks) - get_landmark("right_hip", landmarks)
    # ) / 2.0
    # torso_length = np.linalg.norm(shoulder_center - hip_center)

    # position

    return landmarks


def extract_features(normalized_landmarks):
    """Extract the features we want."""
    result = {}
    selected_landmarks = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        # "left_pinky_1",
        # "right_pinky_1",
        "left_index_1",
        "right_index_1",
        "left_thumb_2",
        "right_thumb_2",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        # "left_heel",
        # "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]
    for lm_name in selected_landmarks:
        lm = get_landmark(lm_name, normalized_landmarks)
        result[f"{lm_name}_x"] = lm[0]
        result[f"{lm_name}_y"] = lm[1]
        result[f"{lm_name}_z"] = lm[2]
    return result


# %% Create dataframe with data for all images
def extract_data(data_dir: str, filetype: str = "png") -> pd.DataFrame:
    """extract poses from all images in given directory. Name of pose is implied from directory structure.

    Args:
        data_dir (str): Directory where to search images.


    Returns:
        pd.DataFrame: Pandas DataFrame with row for each image.
    """
    data = []
    data_dir = Path(data_dir)
    for img_file in data_dir.rglob("*." + filetype):
        # skip checkpint
        if "ipynb_checkpoints" in str(img_file):
            continue

        result = {}
        try:
            #         split by "-", remerge english splits
            names = img_file.parent.name.split("-")
            if len(names) > 1:
                name_sa = names[0].strip()
                name_en = "-".join(names[1:]).strip()
            else:
                name_sa = ""
                name_en = img_file.parent.name.strip()
        except:
            continue

        result["path"] = img_file.relative_to(data_dir)
        result["name_en"] = name_en
        result["name_sa"] = name_sa
        features = extract_features(normalize_body(load_image(str(img_file.resolve()))))
        result.update(features)
        data.append(result)
    return pd.DataFrame(data)


# %% Main
if __name__ == "__main__":
    if len(argv) < 2:
        print(f"Specify path.")
    else:
        df = extract_data(argv[1])
        df.drop("path", axis=1, inplace=True)
        df.to_csv("output.csv")

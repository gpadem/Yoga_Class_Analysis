from pathlib import Path
from sys import argv

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# %% Define helper functions

body_angles = [
    ["nose", "center_shoulders", "center_hips"],
    ["left_hip", "left_shoulder", "left_elbow"],
    ["right_hip", "right_shoulder", "right_elbow"],
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["left_shoulder", "center_shoulders", "nose"],
    ["right_shoulder", "center_shoulders", "nose"],
    ["left_index_1", "left_wrist", "left_elbow"],
    ["right_index_1", "right_wrist", "right_elbow"],
    ["left_shoulder", "left_hip", "left_knee"],
    ["right_shoulder", "right_hip", "right_knee"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_knee", "left_ankle", "left_foot_index"],
    ["right_knee", "right_ankle", "right_foot_index"],
    ["left_elbow", "center_shoulders", "right_elbow"],
    ["left_knee", "center_hips", "right_knee"],
]


def load_image(
    img,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.75,
    **kwargs,
) -> mp.framework.formats.landmark_pb2.NormalizedLandmarkList:
    """All preprocessing task to load a single frame.

    Args:
        img_path (str): Path of the image.
    """
    # load image
    if type(img) in [str, Path]:
        image = cv2.imread(img)
    elif type(img) == np.ndarray:
        image = img
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

    # in case no landmarks were found
    if not landmarks:
        return np.array([np.nan, np.nan, np.nan])

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

    except ValueError:
        if name == "center_hips":
            try:
                return (
                    get_landmark("left_hip", landmarks)
                    + get_landmark("right_hip", landmarks)
                ) / 2
            except:
                return np.array([np.nan, np.nan, np.nan])

        elif name == "center_shoulders":
            try:
                return (
                    get_landmark("left_shoulder", landmarks)
                    + get_landmark("right_shoulder", landmarks)
                ) / 2
            except:
                return np.array([np.nan, np.nan, np.nan])
        else:
            return np.array([np.nan, np.nan, np.nan])


def angle_3d_calculation(
    bodypart1: str, bodypart2: str, bodypart3: str, landmarks_image
):
    """Calculate angle between 3 bodyparts, second one is the vertex."""
    A = get_landmark(bodypart1, landmarks_image)
    B = get_landmark(bodypart2, landmarks_image)
    C = get_landmark(bodypart3, landmarks_image)

    v1 = A - B
    v2 = C - B

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def extract_features(normalized_landmarks):
    """Extract the features we want."""
    result = {}
    selected_landmarks = body_angles
    for body_angle in selected_landmarks:
        angle = angle_3d_calculation(*body_angle, normalized_landmarks)
        feature_name = "_".join(
            ["".join([si.capitalize() for si in s.split("_")]) for s in body_angle]
        )
        result[feature_name] = angle
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
        # skip checkpoint
        if "ipynb_checkpoints" in str(img_file):
            continue

        result = {}
        try:
            if " - " in img_file.parent.name:
                # split by "-", remerge english splits
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
        features = extract_features(load_image(str(img_file.resolve())))
        result.update(features)
        data.append(result)
    return pd.DataFrame(data)


# %% Main
if __name__ == "__main__":
    if len(argv) < 2:
        print(f"Specify path.")
    else:
        df = extract_data(argv[1], "jpg")
        df.drop("path", axis=1, inplace=True)

    output_name = argv[2] if len(argv) > 2 else "output.csv"

    df.to_csv(output_name, index=False)

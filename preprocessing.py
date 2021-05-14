from pathlib import Path
from sys import argv

import cv2
import mediapipe as mp
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


def normalize_body(landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList):
    """Orient the body in the same zay, and normalize distances."""
    return landmarks


def extract_features(normalized_landmarks):
    """Extract the features we want. """
    return {"nose": 0}


# %% Create dataframe with data for all images
def extract_data(data_dir: Path, filetype: str = "png") -> pd.DataFrame:
    """extract poses from all images in given directory. Name of pose is implied from directory structure.

    Args:
        data_dir (Path): Directory where to search/

    Returns:
        pd.DataFrame: Pandas DataFrame with row for each image.
    """
    data = []
    data_dir = Path(data_dir)
    for img_file in data_dir.rglob("*." + filetype):
        result = {}
        try:
            #         split by "-", remerge english splits
            names = img_file.parent.name.split("-")
            if len(names) > 1:
                name_sa = names[0].strip()
                name_en = "-".join(names[1:])
            else:
                name_sa = ""
                name_en = img_file.parent.name.strip()
        except:
            continue

        result["path"] = img_file.relative_to(data_dir)
        result["name_en"] = name_en
        result["name_sa"] = name_sa
        # features = extract_features(normalize_body(load_image(str(d.resolve()))))
        # result.update(features)
        data.append(result)
    return pd.DataFrame(data)


if __name__ == "__main__":
    if len(argv) < 2:
        print(f"Specify path.")
    else:
        df = extract_data(argv[1])

        print(df)

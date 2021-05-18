#  imports
import pickle
import numpy as np
import cv2
import pandas as pd
from preprocessing import extract_features, load_image

#  import RandomForest Model
with open("model.sav", "rb") as f:
    model_dict = pickle.load(f)
    model = [*model_dict.values()][0]


def process_video(file, fps=2):
    cap = cv2.VideoCapture(file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames_between_predict = int(video_fps / fps)

    out = []

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1
        time = counter * 1 / video_fps
        if counter % frames_between_predict != 0:
            continue

        # get body pose
        if ret:
            pose_features = extract_features(load_image(frame))
            predicted_pose = model.predict_proba(
                np.atleast_2d([*pose_features.values()])
            )
            result = {"time": time, "probabilities": predicted_pose}
            for key, val in zip(model.classes_, predicted_pose.flatten()):
                result[key] = val
            out.append(result)

        else:
            break

    cap.release()

    return pd.DataFrame(out)


if __name__ == "__main__":
    test = process_video("DATA/training data/training recording/minute.mp4")
    test.to_csv("test_video_results.csv")

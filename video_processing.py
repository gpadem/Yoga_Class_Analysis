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


def process_video(file, fps=2, show_video=False):
    """Process a video file to extract poses.

    :param file: path to video file
    :param fps: how many frames per second the analysis is done, None for all frames, defult 2
    """
    cap = cv2.VideoCapture(file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames_between_predict = int(video_fps / fps) if fps is not None else 1

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
            if show_video:
                cv2.imshow("Yoga pose detection", frame)
            pose_features = extract_features(load_image(frame))
            predicted_pose = model.predict_proba(
                np.atleast_2d([*pose_features.values()])
            )
            result = {"time": time}
            for key, val in zip(model.classes_, predicted_pose.flatten()):
                result[key] = val
            out.append(result)
            best_pred = get_best_predicition(predicted_pose.flatten(), model.classes_)
            if best_pred:
                print(
                    ", ".join(f"{name} ({prob*100:.0f}%)" for name, prob in best_pred)
                )

        else:
            break

    cap.release()

    return pd.DataFrame(out)


def get_best_predicition(proba, labels, thresholt=0.05) -> list:
    """From a list of predicted pose probabilities, decide the best one(s).

    When the best probabilities are close, more than one result is returned.
    When the resulting list is too long, or the cumulative probability sum is
    less than 50%, nothing is returned.

    :param proba: array of probabilities
    :param labels: array of pose names
    :param thresholt: thesholt percentage, default 0.05
    """
    order = np.argsort(proba)[::-1]
    results = [(labels[order[0]], proba[order[0]])]
    cum_proba = proba[order[0]]
    for o in order[1:]:
        if abs(results[-1][1] - proba[o]) > thresholt:
            break
        results.append((labels[o], proba[o]))
        cum_proba += proba[o]

    if cum_proba > 0.5 and len(results) < 4:
        return results
    else:
        return []


if __name__ == "__main__":
    test = process_video("DATA/training data/training recording/minute.mp4")
    test.to_csv("test_video_results.csv", index=False)

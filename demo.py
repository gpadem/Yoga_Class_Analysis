"""Quick demo."""

from video_processing import process_video
from sys import argv
from pathlib import Path

if __name__ == "__main__":
    print("Yoga pose Demo.")
    if len(argv) == 1:
        model = ""
        while model not in ["nn", "rf"]:
            model = input("Random forest [rf] or Forward-feeding NN [nn]? ")

        method = ""
        while method not in ["w", "v"]:
            method = input("[w]ebcam or [v]ideo file? ")
        if method == "w":
            process_video(0, fps=3, model_choice=model, show_video=True)
        elif method == "v":
            filename = Path(input("Specify video file path. "))
            if filename.exists():
                process_video(
                    str(filename.resolve()), fps=3, model_choice=model, show_video=True
                )
            else:
                f"File {filename} does not exist."

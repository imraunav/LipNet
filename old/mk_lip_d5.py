import numpy as np
import cv2
import os
import dlib
from tqdm import tqdm
import pickle

from preprocessing import vidread
from utils import LipDetector


def main(path):
    vid_path = os.path.join(path, "videos")
    frames_path = os.path.join(path, "frames")
    lipextractor = LipDetector()
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("./dlib_dat/shape_predictor_68_face_landmarks.dat")
    extra = 10

    for spk in sorted(os.listdir(vid_path))[4::5]:
        print(f"Extracting speaker {spk}...")
        if spk == ".DS_Store":
            continue
        spk_vid_path = os.path.join(vid_path, spk)  # speaker path
        spk_frames_path = os.path.join(frames_path, spk)

        if not os.path.exists(spk_frames_path):
            os.makedirs(spk_frames_path)

        for vidfile in tqdm(os.listdir(spk_vid_path)):
            if ".mpg" not in vidfile:
                continue

            vidfile_path = os.path.join(spk_vid_path, vidfile)
            framefile_path = os.path.join(
                spk_frames_path, vidfile.split(".")[0] + ".pkl"
            )

            frames = vidread(vidfile_path)
            lips = []
            for frame in frames:
                lip = lipextractor.findlip(frame)
                if lip is not None:
                    lip = cv2.resize(lip, (100, 50))
                    lips.append(lip)
                with open(framefile_path, mode='wb') as f:
                    pickle.dump(lips, f, pickle.HIGHEST_PROTOCOL)
        print(f"Done for speaker {spk}.")


if __name__ == "__main__":
    main(path="dataset/test")
    main(path="dataset/train")

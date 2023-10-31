import numpy as np
import cv2
import os
from preprocessing import vidread
import dlib
from tqdm import tqdm

# from utils import LipDetector


def main():
    path = "dataset/test"
    vid_path = os.path.join(path, "videos")
    # lipextractor = LipDetector()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./dlib_dat/shape_predictor_68_face_landmarks.dat")
    extra = 10

    for spk in os.listdir(vid_path):
        frame_path = os.path.join(path, "frames", spk)
        os.makedirs(frame_path, exist_ok=True)
        for fname in tqdm(os.listdir(os.path.join(vid_path, spk))):
            if ".mpg" not in fname:
                continue

            fpath = os.path.join(vid_path, spk, fname)
            os.makedirs(os.path.join(frame_path, fname.split(".")[0]), exist_ok=True)
            vidframes = vidread(fpath)
            frameno = 0
            for frame in vidframes:
                frameno += 1
                grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = detector(grey)
                if faces:
                    # Assuming there's only one face detected
                    face = faces[0]

                    # Detect facial landmarks
                    landmarks = predictor(grey, face)

                    # Extract the lip region using landmarks
                    lip_x = landmarks.part(48).x  # Left corner of the mouth
                    lip_y = landmarks.part(51).y  # Top of the upper lip
                    lip_w = (
                        landmarks.part(54).x - landmarks.part(48).x
                    )  # Width of the mouth
                    lip_h = (
                        landmarks.part(57).y - landmarks.part(51).y
                    )  # Height of the upper lip

                    # Extract the lip region
                    lip_region = frame[
                        lip_y - extra : lip_y + extra + lip_h,
                        lip_x - extra : lip_x + extra + lip_w,
                    ]

                    lip_region = cv2.cvtColor(lip_region, cv2.COLOR_RGB2BGR)
                    # print(frame_path, fname.split(".")[0], str(frameno) + ".png")
                    cv2.imwrite(
                        os.path.join(
                            frame_path, fname.split(".")[0], str(frameno) + ".png"
                        ),
                        lip_region,
                    )
                    # print(
                    #     os.path.join(
                    #         frame_path, fname.split(".")[0], str(frameno) + ".png"
                    #     )
                    # )
                    # cv2.imshow("Lip", lip_region)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit()
                else:
                    print(frame_path, fname.split(".")[0])
                    print("No face detected in the image.")


if __name__ == "__main__":
    main()

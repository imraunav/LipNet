import dlib
import cv2
import numpy as np
import pickle
import editdistance


class TokenConv:
    """
    0 key is the invalid key for the network, hence staring everything from 1
    """

    def __init__(self):
        oov_char = ""  # out of vocab char
        self.vocab = " abcdefghijklmnopqrstuvwxyz"
        self.char2int = {c: i for i, c in enumerate(self.vocab, start=1)}
        self.int2char = {i: c for i, c in enumerate(self.vocab, start=1)}

        self.char2int[oov_char] = 0
        self.int2char[0] = oov_char

    def encode(self, charecters: list) -> list[str]:
        return [self.char2int.get(c, "") for c in charecters]

    # def decode(self, ints: list) -> list:
    #     return [self.int2char.get(i) for i in ints]  # __default= ?

    def ctc_decode(self, arr: list) -> str:
        decoded_sequence = []
        previous_label = None
        for x in arr:
            if x != previous_label:  # oov_char taken care of
                decoded_sequence.append(self.int2char.get(x))
            previous_label = x
        return "".join(decoded_sequence)


def HorizontalFlip(frames, p=0.5) -> list[np.array]:
    # frames = np.array(batch_img)  # convenience
    # (T, W, H, C)
    if np.random.random() > p:
        frames = [frame[:, ::-1, ...] for frame in frames]
    return frames


def vidread(filepath) -> list[np.array]:
    vid_cap = cv2.VideoCapture(filepath)
    frames = []
    while vid_cap.isOpened() == True:
        ret, frame = vid_cap.read()
        if ret == True:
            # frame = frame[190:236, 80:220, :] # crop to lip region
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    vid_cap.release()
    return frames


class LipDetector:
    """
    A class to find lip in a face image
    http://dlib.net/files/
    """

    def __init__(self):
        # Load the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./dlib_dat/shape_predictor_68_face_landmarks.dat"
        )

    def findlip(self, im, extra=10):
        # Resize to get better chance of finding a face
        # im = cv2.resize(im, dsize=(1000, 1000))
        # Detect faces in the grayscale image
        grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        faces = self.detector(grey)
        if faces:
            # Assuming there's only one face detected
            face = faces[0]

            # Detect facial landmarks
            landmarks = self.predictor(grey, face)

            # Extract the lip region using landmarks
            lip_x = landmarks.part(48).x  # Left corner of the mouth
            lip_y = landmarks.part(51).y  # Top of the upper lip
            lip_w = landmarks.part(54).x - landmarks.part(48).x  # Width of the mouth
            lip_h = (
                landmarks.part(57).y - landmarks.part(51).y
            )  # Height of the upper lip

            # Extract the lip region
            lip_region = im[
                lip_y - extra : lip_y + extra + lip_h,
                lip_x - extra : lip_x + extra + lip_w,
            ]
            return lip_region

        else:
            # np.zeros((100, 100, 3)) # dummy blank frame
            return None
            # print("No face detected in the image.")


def get_frames_pkl(path) -> list[np.array]:
    with open(path, mode="rb") as f:
        frames = pickle.load(f)
    return frames


def load_align(p) -> list[str]:
    with open(p, "r") as file:
        lines = file.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != "sil":  # ignore if silence
            tokens.extend(list(line[2]))  # only add the words as chars
            tokens.append(" ")
    return tokens[:-1]  # remove last space


def padding(array, length):
    array = np.array(array)  # convenience
    array = [array[_] for _ in range(array.shape[0])]
    size = array[0].shape
    for i in range(length - len(array)):
        array.append(np.zeros(size))
    return np.stack(array, axis=0)


def wer(predict, truth) -> list:
    word_pairs = [(p[0].split(" "), p[1].split(" ")) for p in zip(predict, truth)]
    wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
    return wer


def cer(predict, truth):
    cer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in zip(predict, truth)]
    return cer

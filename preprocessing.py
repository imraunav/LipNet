import dlib
import cv2
import numpy as np

class CTCCoder:

    def __init__(self, start = 1):
        self.vocab = list(" abcdefghijklmnopqrstuvwxyz")
        self.char2int = {c: i for i, c in enumerate(self.vocab, start=start)}
        self.int2char = {i: c for i, c in self.char2int.items()}

    def encode_char(self, charecter:list):
        return [self.char2int[str(x)] for x in charecter] # just a safeguard, added str

    def decode_char(self, integer:list):
        return [self.int2char[int(x)] for x in integer] # just a safeguard, added int
    
    def ctc_arr2txt(self, arr, start=1):
        # self.int2char = {i+start : x for i, x in enumerate(' abcdefghijklmnopqrstuvwxyz')}
        prev = -1
        txt = []
        for n in arr:
            if(prev != n and n >= start):
                if(len(txt) > 0 and txt[-1] == ' ' and self.int2char[n] == ' '):
                    pass
                else:
                    txt.append(self.int2char[int(n)]) # just a safeguard, added int
            prev = n
        return ''.join(txt).strip()

def HorizontalFlip(batch_img, p=0.5):
    batch_img = np.array(batch_img) # convenience
    # (T, H, W, C)
    if np.random.random() > p:
        batch_img = batch_img[:, :, ::-1, ...]
    return batch_img


def vidread(filepath):
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
        self.predictor = dlib.shape_predictor("./dlib/shape_predictor_68_face_landmarks.dat")

    def findlip(self, im, extra=10):
        # Detect faces in the grayscale image
        grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        faces = self.detector(im)
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
            print("No face detected in the image.")

import torch
from torch.utils.data import Dataset, DataLoader
import os
from preprocessing import vidread, LipDetector, HorizontalFlip, CTCCoder
import cv2
import numpy as np
import editdistance

class LipDataset(Dataset):
    def __init__(self, dataset_path, vid_pad=75, align_pad=40, phase="train"):
        """
        Dataset path to entire dataset, find location of all the files within that directory
        """
        self.align_path = os.path.join(dataset_path, phase, "alignments")
        self.vid_path = os.path.join(dataset_path, phase, "videos")
        self.vid_pad = vid_pad
        self.align_pad = align_pad
        self.phase = phase

        self.ctccoder = CTCCoder()  # encode and decode alignment to ctc format

        self.data = []
        for path, subdirs, files in os.walk(self.vid_path):
            if len(subdirs) != 0:  # if not in subdir, don't do anything
                continue

            spk = path.split(os.path.sep)[-1]  # only speaker name from path
            # print("Speaker: ", spk)
            
            for file in files:
                if ".mpg" not in file:  # skip non-video files
                    continue
            
                # print((spk, file.split(".")[0]))
            
                fname = file.split(".")[0]  # only name of the file without extention
                if os.path.exists(os.path.join(self.align_path, spk, fname+'.align')) == True: # only add when the alignment also exists
                    self.data.append((spk, fname))  # speaker-name and name of the file
        return None

    def __getitem__(self, idx):
        """
        Return a dictionary with key values:
        "vid", "align", "align_len", "vid_len"
        """
        spk, fname = self.data[idx]
        vid_path = os.path.join(self.vid_path, spk, fname + ".mpg")
        align_path = os.path.join(self.align_path, spk, fname + ".align")

        vid = self._load_video(vid_path, lip_size=(100, 50))
        align = self._load_align(align_path)

        if self.phase == "train":
            vid = HorizontalFlip(vid)

        vid_len = len(vid)
        align_len = len(align)
        vid = self._padding(vid, self.vid_pad)
        align = self._padding(align, self.align_pad)

        return {
            "vid": torch.Tensor(vid),
            "align": torch.Tensor(align),
            "align_len": align_len,
            "vid_len": vid_len,
        }

    def __len__(self):
        return len(self.data)

    def _load_video(self, p, lip_size=(64, 32)):
        # read video frames
        frames = vidread(p)
        # extract lips from each frame
        lipextractor = LipDetector()
        lips = []
        for f in frames:
            lip = lipextractor.findlip(f)
            # resize each lip frame to same size
            if lip is None:
                lip = np.zeros(*lip_size, 3)
            else:
                lip = cv2.resize(lip, dsize=lip_size)
            lips.append(lip)
        return lips

    def _load_align(self, p):
        with open(p, "r") as file:
            lines = file.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != "sil":  # ignore if silence
                tokens.append(" ")
                tokens.extend(list(line[2]))  # only add the words as chars

        return self.ctccoder.encode_char(tokens)

    def _padding(self, array, length):
        array = np.array(array) # convenience
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    def wer(self, predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    def cer(self, predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer

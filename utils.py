import torch
from torch.utils.data import Dataset
import os
from preprocessing import HorizontalFlip, get_frames_pkl, load_align, TokenConv, padding
import numpy as np


class LipDataset(Dataset):
    def __init__(self, dataset_path, vid_pad=75, align_pad=40, phase="train") -> None:
        super().__init__()
        self.align_path = os.path.join(dataset_path, phase, "alignments")
        # self.vid_path = os.path.join(dataset_path, phase, "videos")
        self.frames_path = os.path.join(dataset_path, phase, "frames")
        self.vid_pad = vid_pad
        self.align_pad = align_pad
        self.phase = phase
        self.ctccoder = TokenConv()

        self.data = []
        for path, subdirs, files in os.walk(self.frames_path):
            if len(subdirs) != 0:  # if not in subdir, don't do anything
                continue

            spk = path.split(os.path.sep)[-1]  # only speaker name from path
            # print("Speaker: ", spk)

            for file in files:
                # if ".mpg" not in file:  # skip non-video files
                #     continue
                if ".pkl" not in file:  # skip non-pickle files
                    continue
                # print((spk, file.split(".")[0]))

                fname = file.split(".")[0]  # only name of the file without extention
                align_dir = os.path.join(self.align_path, spk, fname + ".align")
                if os.path.exists(align_dir):  # only add when the alignment also exists
                    self.data.append((spk, fname))  # speaker-name and name of the file
        print("Dataset loaded successfully!")
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        speaker, fname = self.data[index]
        frames_path = os.path.join(self.frames_path, speaker, fname + ".pkl")
        align_path = os.path.join(self.align_path, speaker, fname + ".align")

        vid = get_frames_pkl(frames_path)
        align = load_align(align_path)
        align = self.ctccoder.encode(align)

        if self.phase == "train":
            vid = HorizontalFlip(vid)

        vid_len = len(vid)
        align_len = len(align)
        vid = padding(vid, self.vid_pad)
        align = padding(align, self.align_pad)

        return (
            torch.Tensor(vid),
            torch.Tensor(align),
            vid_len,
            align_len,
        )

# %%
# import os

# train = "./dataset/train/videos"

# for path, dirs, files in os.walk(train):
#     print(path) # returns the paths in the folder(speaker paths)
#     # # print(dirs) # All available subfolders (only folders)
#     # print(files) # all files the corresponding path()
#     # print(len(dirs))


# %%
# from utils import LipDataset

# train = "./dataset"
# dataset = LipDataset(train)

# # print(len(dataset))
# print(dataset.__getitem__(23)['txt_len'])
# print(dataset.__getitem__(37)['txt'].shape)
# %%
# %%
# from utils import LipDataset

# train = "./dataset"
# dataset = LipDataset(train)

# # print(len(dataset))
# print(dataset.__getitem__(23)['align_len'])
# print(dataset.__getitem__(37)['align'])

# %%
# from preprocessing import vidread, LipDetector
# import cv2
# import numpy as np

# vidpath = './dataset/test/videos/s2/pbib7p.mpg'
# frames = vidread(vidpath)
# lips = []
# lipextractor = LipDetector()
# def _load_video(path, lip_size=(100, 50)):
#     # read video frames
#     frames = vidread(path)
#     # extract lips from each frame
#     lipextractor = LipDetector()
#     lips = []
#     for f in frames:
#         lip = lipextractor.findlip(f)
#         if lip is not None:
#             lip = cv2.resize(lip, lip_size)
#             lips.append(lip)
#     return np.array(lips)
# for frame in frames:
#     lip = lipextractor.findlip(frame)
#     if lip is not None:
#         lip = cv2.resize(lip, (100, 50))
#         lips.append(lip)
# print(len(lips))
# lips = _load_video(vidpath)
# for fno, lip in enumerate(lips):
#     cv2.imshow(f'Frame: {fno}', lip[:,:,::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# %%
# from utils import LipDataset

# path = "./dataset"
# ds = LipDataset(path)
# # print(ds.data)
# for i in range(len(ds)):
#     d = ds.__getitem__(i)
#     print(d['vid'].shape)

# %%
import pickle
import cv2

PATH = "./dataset/train/frames/s3/bbaf1s.pkl"
with open(PATH, mode='rb') as f:
    frames = pickle.load(f)

for i, frame in enumerate(frames):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow(f"Frame {i+1}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cv2.destroyAllWindows()
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
from utils import LipDataset

train = "./dataset"
dataset = LipDataset(train)

# print(len(dataset))
print(dataset.__getitem__(23)['align_len'])
print(dataset.__getitem__(37)['align'])
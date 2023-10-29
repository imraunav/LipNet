# train_speakers = (1, 2, 3, 4, 5, 6, 7, 8, 9, )
# test_speakers = (1, 2, 20, 22)
dataset_path = "./dataset"
phase = "train"
vid_pad = 75
align_pad = 40
batch_size = 96
num_workers = 16 # what is this?
learning_rate = 2e-5
max_epoch = 10_000
gpu = '0'
display = 10
data_type = 'unseen'
save_prefix = f'weights/LipNet_{data_type}'
is_optimize = True


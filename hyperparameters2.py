# train_speakers = (1, 2, 3, 4, 5, 6, 7, 8, 9, )
# test_speakers = (1, 2, 20, 22)
dataset_path = "./dataset"
phase = "train"
vid_pad = 75
align_pad = 200
batch_size = 96//2 #25 #50 #50
num_workers = 8  # what is this?, number of cores used to load data
base_learning_rate = 2e-5
start_epoch = 0
max_epoch = 10_000
gpu = "0,1"
display = 50
data_type = "unseen"
save_prefix = f"weights/LipNet_{data_type}"
is_optimize = True
test_step = 1000
num_gpus = 2
save_every = 100
debug = False

# weights = "weights/lipnet_900_wer:0.3242.pt"

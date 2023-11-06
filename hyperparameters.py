# train_speakers = (1, 2, 3, 4, 5, 6, 7, 8, 9, )
# test_speakers = (1, 2, 20, 22)
dataset_path = "./dataset"
phase = "train"
vid_pad = 75
align_pad = 200
batch_size = 96
num_workers = 16  # what is this?, number of cores used to load data
base_learning_rate = 2e-5
max_epoch = 2
gpu = "0,1"
display = 10
data_type = "unseen"
save_prefix = f"weights/LipNet_{data_type}"
is_optimize = True
test_step = 1000
num_gpus = 2
save_every = 100
debug = True

weights = "weights/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt"

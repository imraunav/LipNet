import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

# from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

from model import LipNet
from utils import LipDataset
from preprocessing import wer, cer, TokenConv
import hyperparameters
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def train(model, device):
    # load hyperparameters
    path = hyperparameters.dataset_path
    vid_pad = hyperparameters.vid_pad
    align_pad = hyperparameters.align_pad
    phase = hyperparameters.phase
    batch_size = hyperparameters.batch_size
    num_workers = hyperparameters.num_workers
    base_learning_rate = hyperparameters.base_learning_rate
    max_epoch = hyperparameters.max_epoch
    if phase == "train":
        shuffle = True
    else:
        shuffle = False

    # instantiate dataset and dataloader
    dataset = LipDataset(path, vid_pad, align_pad, phase)
    sampler = DistributedSampler(
        dataset,
        num_replicas=hyperparameters.num_gpus,
    )
    dataloader = DataLoader(
        dataset, batch_size, shuffle, num_workers=num_workers, sampler=sampler
    )

    optimizer = optim.Adam(model.parameters(), lr=base_learning_rate, amsgrad=True)
    ctc = nn.CTCLoss()
    ctcdecoder = TokenConv()
    tic = time.time()

    train_wer = []
    for epoch in range(max_epoch):
        print("Epoch : ", epoch)
        for i, (vid, align, vid_len, align_len) in enumerate(dataloader):
            vid = vid.to(device)
            align = align.to(device)
            vid_len = vid_len.to(device)
            align_len = align_len.to(device)

            y = model(vid)
            loss = ctc(
                y.transpose(0, 1),
                align,
                vid_len.view(-1),
                align_len.view(-1),
            )
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y = torch.argmax(y, dim=2)
            for tru, pre in zip(align.tolist(), y.tolist()):
                true_txt = ctcdecoder.ctc_decode(tru)
                pred_txt = ctcdecoder.ctc_decode(pre)

                train_wer.extend(wer(pred_txt, true_txt))
                if epoch % hyperparameters.display:
                    print("True: ", true_txt)
                    print("Pred: ", pred_txt)
        if epoch % hyperparameters.display:
            torch.save(
                model.state_dict(),
                f"./weights/lipnet_{epoch}_wer:{np.mean(train_wer):.4f}.pt",
            )


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LipNet()  # instantiate model
    model = DistributedDataParallel(model).to(device)
    train(model, device)


if __name__ == "__main__":
    # writer = SummaryWriter()
    dist.init_process_group(backend="nccl", init_method="env://")
    main()
    dist.destroy_process_group()

# def test(model, net):
#     path = hyperparameters.dataset_path
#     vid_pad = hyperparameters.vid_pad
#     align_pad = hyperparameters.align_pad
#     phase = "test"
#     shuffle = False
#     batch_size = hyperparameters.batch_size
#     num_workers = hyperparameters.num_workers
#     base_learning_rate = hyperparameters.base_learning_rate
#     max_epoch = hyperparameters.max_epoch

#     with torch.no_grad():
#         dataset = LipDataset(path, vid_pad, align_pad, phase)
#         dataloader = DataLoader(dataset, batch_size, shuffle, num_workers)

#         print("num_test_data:{}".format(len(dataset.data)))
#         model.eval()
#         loss_list = []
#         wer = []
#         cer = []
#         crit = nn.CTCLoss()
#         tic = time.time()
#         for i_iter, data in enumerate(dataloader):
#             vid = data.get("vid").cuda()
#             txt = data.get("txt").cuda()
#             vid_len = data.get("vid_len").cuda()
#             txt_len = data.get("txt_len").cuda()

#             y = net(vid)

#             loss = (
#                 crit(
#                     y.transpose(0, 1).log_softmax(-1),
#                     txt,
#                     vid_len.view(-1),
#                     txt_len.view(-1),
#                 )
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )
#             loss_list.append(loss)
#             pred_txt = prediction(y)

#             ctcdecoder = CTCCoder()
#             truth_txt = [ctcdecoder.ctc_arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
#             wer.extend(dataset.wer(pred_txt, truth_txt))
#             cer.extend(dataset.cer(pred_txt, truth_txt))
#             if i_iter % hyperparameters.display == 0:
#                 v = 1.0 * (time.time() - tic) / (i_iter + 1)
#                 eta = v * (len(dataloader) - i_iter) / 3600.0

#                 print("".join(101 * "-"))
#                 print("{:<50}|{:>50}".format("predict", "truth"))
#                 print("".join(101 * "-"))
#                 for predict, truth in list(zip(pred_txt, truth_txt))[:10]:
#                     print("{:<50}|{:>50}".format(predict, truth))
#                 print("".join(101 * "-"))
#                 print(
#                     "test_iter={},eta={},wer={},cer={}".format(
#                         i_iter, eta, np.array(wer).mean(), np.array(cer).mean()
#                     )
#                 )
#                 print("".join(101 * "-"))

#         return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

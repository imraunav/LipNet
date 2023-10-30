import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

from model import LipNet
from utils import LipDataset
from preprocessing import CTCCoder
import hyperparameters


def train(model, net, device):
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
    dataloader = DataLoader(dataset, batch_size, shuffle)

    optimizer = optim.Adam(model.parameters(), lr=base_learning_rate, amsgrad=True)
    ctc = nn.CTCLoss()
    ctcdecoder = CTCCoder()
    tic = time.time()

    train_wer = []
    for epoch in range(max_epoch):
        print("Epoch: ", epoch)
        i_iter = 0
        for data in dataloader:
            i_iter += 1
            model.train()
            # dataset.__getitem__()
            vid = data.get("vid").to(device)
            align = data.get("align").to(device)
            vid_len = data.get("vid_len").to(device)
            align_len = data.get("align_len").to(device)

            optimizer.zero_grad()
            y = net(vid)

            loss = ctc(
                y.transpose(0, 1).log_softmax(-1),
                align,
                vid_len.view(-1),
                align_len.view(-1),
            )
            loss.backward()
            optimizer.step()

            tot_iter = i_iter + epoch * len(dataloader)

            pred_txt = prediction(y)

            truth_txt = [
                ctcdecoder.ctc_arr2txt(align[_], start=1) for _ in range(align.size(0))
            ]
            # print(len(pred_txt))
            # print(len(truth_txt))
            train_wer.extend(dataset.wer(pred_txt, truth_txt))

            if tot_iter % hyperparameters.display == 0:
                v = 1.0 * (time.time() - tic) / (tot_iter + 1)
                eta = (len(dataloader) - i_iter) * v / 3600.0

                writer.add_scalar("train loss", loss, tot_iter)
                writer.add_scalar("train wer", np.array(train_wer).mean(), tot_iter)
                print("".join(101 * "-"))
                print("{:<50}|{:>50}".format("predict", "truth"))
                print("".join(101 * "-"))

                for predict, truth in list(zip(pred_txt, truth_txt))[:3]:
                    print("{:<50}|{:>50}".format(predict, truth))
                print("".join(101 * "-"))
                print(
                    "epoch={},tot_iter={},eta={},loss={},train_wer={}".format(
                        epoch, tot_iter, eta, loss, np.array(train_wer).mean()
                    )
                )
                print("".join(101 * "-"))

            if tot_iter % hyperparameters.test_step == 0:
                (loss, wer, cer) = test(model, net)
                print(
                    "i_iter={},lr={},loss={},wer={},cer={}".format(
                        tot_iter, show_lr(optimizer), loss, wer, cer
                    )
                )
                writer.add_scalar("val loss", loss, tot_iter)
                writer.add_scalar("wer", wer, tot_iter)
                writer.add_scalar("cer", cer, tot_iter)
                savename = "{}_loss_{}_wer_{}_cer_{}.pt".format(
                    hyperparameters.save_prefix, loss, wer, cer
                )
                (path, name) = os.path.split(savename)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if not hyperparameters.is_optimize:
                    exit()


def prediction(y):
    result = []
    y = y.argmax(-1)
    ctccoder = CTCCoder()
    return [ctccoder.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group["lr"]]
    return np.array(lr).mean()


def test(model, net):
    path = hyperparameters.dataset_path
    vid_pad = hyperparameters.vid_pad
    align_pad = hyperparameters.align_pad
    phase = "test"
    shuffle = False
    batch_size = hyperparameters.batch_size
    num_workers = hyperparameters.num_workers
    base_learning_rate = hyperparameters.base_learning_rate
    max_epoch = hyperparameters.max_epoch

    with torch.no_grad():
        dataset = LipDataset(path, vid_pad, align_pad, phase)
        dataloader = DataLoader(dataset, batch_size, shuffle, num_workers)

        print("num_test_data:{}".format(len(dataset.data)))
        model.eval()
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for i_iter, data in enumerate(dataloader):
            vid = data.get("vid").cuda()
            txt = data.get("txt").cuda()
            vid_len = data.get("vid_len").cuda()
            txt_len = data.get("txt_len").cuda()

            y = net(vid)

            loss = (
                crit(
                    y.transpose(0, 1).log_softmax(-1),
                    txt,
                    vid_len.view(-1),
                    txt_len.view(-1),
                )
                .detach()
                .cpu()
                .numpy()
            )
            loss_list.append(loss)
            pred_txt = prediction(y)

            ctcdecoder = CTCCoder()
            truth_txt = [ctcdecoder.ctc_arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(dataset.wer(pred_txt, truth_txt))
            cer.extend(dataset.cer(pred_txt, truth_txt))
            if i_iter % hyperparameters.display == 0:
                v = 1.0 * (time.time() - tic) / (i_iter + 1)
                eta = v * (len(dataloader) - i_iter) / 3600.0

                print("".join(101 * "-"))
                print("{:<50}|{:>50}".format("predict", "truth"))
                print("".join(101 * "-"))
                for predict, truth in list(zip(pred_txt, truth_txt))[:10]:
                    print("{:<50}|{:>50}".format(predict, truth))
                print("".join(101 * "-"))
                print(
                    "test_iter={},eta={},wer={},cer={}".format(
                        i_iter, eta, np.array(wer).mean(), np.array(cer).mean()
                    )
                )
                print("".join(101 * "-"))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LipNet()  # instantiate model
    model.to(device)
    net = nn.DataParallel(model).to(device)
    train(model, net, device)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = hyperparameters.gpu
    writer = SummaryWriter()
    main()

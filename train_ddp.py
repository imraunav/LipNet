import os
import torch
from torch import nn, optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import numpy as np

import hyperparameters
from preprocessing import TokenConv, wer
from utils import LipDataset
from model import LipNet


# Each process control a single gpu
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(
    trainset: Dataset,
    bs: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    sampler_train = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=False,
        sampler=sampler_train,
        num_workers=hyperparameters.num_workers,
    )

    return trainloader, sampler_train


class TrainerDDP:
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        trainloader: DataLoader,
        sampler_train: DistributedSampler,
    ) -> None:
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.model = DDP(model, device_ids=[gpu_id])
        self.trainloader = trainloader

        self.sampler_train = sampler_train

        self.optimizer = optim.Adam(
            model.parameters(), lr=hyperparameters.base_learning_rate, amsgrad=True
        )
        self.crit = nn.CTCLoss()
        self.ctcdecoder = TokenConv()

    def _save_checkpoint(self, epoch: int, train_wer):
        ckp = self.model.module.state_dict()
        model_path = f"./weights/lipnet_{epoch}_wer:{np.mean(train_wer):.4f}.pt.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        train_wer = []
        for epoch in range(max_epochs):
            epoch_loss = 0
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            self.sampler_train.set_epoch(epoch)
            for i, (vid, align, vid_len, align_len) in enumerate(self.trainloader):
                vid = vid.to(self.gpu_id)
                align = align.to(self.gpu_id)
                vid_len = vid_len.to(self.gpu_id)
                align_len = align_len.to(self.gpu_id)
                y = self.model(vid)
                loss = self.crit(
                    y.transpose(0, 1),
                    align,
                    vid_len.view(-1),
                    align_len.view(-1),
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                y = torch.argmax(y, dim=2)
            for tru, pre in zip(align.tolist(), y.tolist()):
                true_txt = self.ctcdecoder.ctc_decode(tru)
                pred_txt = self.ctcdecoder.ctc_decode(pre)

                train_wer.extend(wer(pred_txt, true_txt))
                if epoch % hyperparameters.display:
                    print(f"Epoch [GPU:{self.gpu_id}]: ")
                    print("True: ", true_txt)
                    print("Pred: ", pred_txt)
            print(
                f"Epoch [GPU:{self.gpu_id}]: ",
                epoch,
                "Loss : ",
                epoch_loss / len(self.trainloader),
            )
            # only save once on master gpu
            if self.gpu_id == 0 and epoch % hyperparameters.save_every == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(max_epochs - 1)


def main(rank, world_size):
    ddp_setup(rank, world_size)  # initialize ddp

    train_dataset = LipDataset(hyperparameters.dataset_path)
    train_dataloader, train_sampler = dataloader_ddp(
        train_dataset, hyperparameters.batch_size
    )
    model = LipNet()
    trainer = TrainerDDP(
        gpu_id=rank,
        model=model,
        trainloader=train_dataloader,
        sampler_train=train_sampler,
    )
    trainer.train(hyperparameters.max_epoch)
    destroy_process_group()  # clean up


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  # nprocs - total number of processes - # gpus

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 5, 5),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 5, 5),
            stride=(1, 1, 1),
            padding=(1, 2, 2),
        )
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=96,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.gru1 = nn.GRU(
            input_size=96 * 3 * 6, hidden_size=256, num_layers=1, bidirectional=True
        )  # 96 * 4 * 8
        # 2 because of bi-direction
        self.gru2 = nn.GRU(
            input_size=256 * 2, hidden_size=256, num_layers=1, bidirectional=True
        )
        self.linear = nn.Linear(256 * 2, 27 + 1)

    def forward(self, x):
        # normalize data first
        x = x / 255.0

        # (B, T, H, W, C)->(B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        # STCNN block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout3d(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # STCNN block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout3d(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # STCNN block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout3d(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()  # GPU and pytorch idiosyncrasy
        self.gru2.flatten_parameters()
        # print(x.shape)
        x, _ = self.gru1(x)
        x = F.dropout(x)
        x, _ = self.gru2(x)
        x = F.dropout(x)

        x = self.linear(x)
        # print("Output of linear layer shape: ", x.shape)
        # (T, B, 27+1) -> (B, T, 27+1)
        x = x.permute(1, 0, 2).contiguous()
        return F.log_softmax(x, dim=-1)  # use torch.exp() to get softmax values

    def predict(self, frames: list):
        in_frames = np.array(frames)
        if len(in_frames.shape) != 5:  # batch when needed
            in_frames = in_frames.reshape((1, *in_frames.shape))
        # in_frames.shape
        out_pred = self.forward(torch.Tensor(in_frames))  # pass through model
        out_pred = torch.argmax(out_pred, dim=2)  # one-hot to numbers
        return out_pred

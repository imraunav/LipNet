import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import init
import math


class LipNet(nn.Module):
    def __init__(self, vocab_size=27):
        super(LipNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)
            ),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(),
            nn.Conv3d(
                32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)
            ),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(),
            nn.Conv3d(
                64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            ),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(),
        )
        # T B C*H*W
        self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=True)
        self.drp1 = nn.Dropout()
        # T B F
        self.gru2 = nn.GRU(256 * 2, 256, 1, bidirectional=True)
        self.drp2 = nn.Dropout()
        # T B V
        self.pred = nn.Linear(256 * 2, vocab_size + 1)

        # initialisations
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity="relu")
                init.constant_(m.bias, 0)

        init.kaiming_normal_(self.pred.weight, nonlinearity="sigmoid")
        init.constant_(self.pred.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv(x)  # B C T H W
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # T B C H W
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x, _ = self.gru2(x)
        x = self.drp2(x)
        x = x.permute(1, 0, 2).contiguous() # (T, B, 27+1) -> (B, T, 27+1)
        x = self.pred(x)

        return F.log_softmax(x, dim=-1)


# class LipNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv3d(
#             in_channels=3,
#             out_channels=32,
#             kernel_size=(3, 5, 5),
#             stride=(1, 2, 2),
#             padding=(1, 2, 2),
#         )
#         self.conv2 = nn.Conv3d(
#             in_channels=32,
#             out_channels=64,
#             kernel_size=(3, 5, 5),
#             stride=(1, 1, 1),
#             padding=(1, 2, 2),
#         )
#         self.conv3 = nn.Conv3d(
#             in_channels=64,
#             out_channels=96,
#             kernel_size=(3, 3, 3),
#             stride=(1, 1, 1),
#             padding=(1, 1, 1),
#         )
#         self.gru1 = nn.GRU(
#             input_size=96 * 3 * 6, hidden_size=256, num_layers=1, bidirectional=True
#         )  # 96 * 4 * 8
#         # 2 because of bi-direction
#         self.gru2 = nn.GRU(
#             input_size=256 * 2, hidden_size=256, num_layers=1, bidirectional=True
#         )
#         self.linear = nn.Linear(256 * 2, 27 + 1)

#     def forward(self, x):
#         # normalize data first
#         x = x / 255.0

#         # (B, T, H, W, C)->(B, C, T, H, W)
#         x = x.permute(0, 4, 1, 2, 3)

#         # STCNN block 1
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.dropout3d(x)
#         x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

#         # STCNN block 2
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.dropout3d(x)
#         x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

#         # STCNN block 3
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = F.dropout3d(x)
#         x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

#         # (B, C, T, H, W)->(T, B, C, H, W)
#         x = x.permute(2, 0, 1, 3, 4).contiguous()
#         # (T, B, C, H, W)->(T, B, C*H*W)
#         x = x.view(x.size(0), x.size(1), -1)

#         self.gru1.flatten_parameters()  # GPU and pytorch idiosyncrasy
#         self.gru2.flatten_parameters()
#         # print(x.shape)
#         x, _ = self.gru1(x)
#         x = F.dropout(x)
#         x, _ = self.gru2(x)
#         x = F.dropout(x)

#         x = self.linear(x)
#         # print("Output of linear layer shape: ", x.shape)
#         # (T, B, 27+1) -> (B, T, 27+1)
#         x = x.permute(1, 0, 2).contiguous()
#         return F.log_softmax(x, dim=-1)  # use torch.exp() to get softmax values

#     def predict(self, frames: list):
#         in_frames = np.array(frames)
#         if len(in_frames.shape) != 5:  # batch when needed
#             in_frames = in_frames.reshape((1, *in_frames.shape))
#         # in_frames.shape
#         out_pred = self.forward(torch.Tensor(in_frames))  # pass through model
#         out_pred = torch.argmax(out_pred, dim=2)  # one-hot to numbers
#         return out_pred

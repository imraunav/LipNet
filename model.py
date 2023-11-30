import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import init
import math


class LipNet(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

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
        # (B, T, H, W, C)->(B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return F.log_softmax(x, dim=-1)


class LipNet_uni(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=False)
        self.gru2 = nn.GRU(256, 256, 1, bidirectional=False)

        self.FC = nn.Linear(256, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

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
                # init.uniform_(
                #     m.weight_ih_l0_reverse[i : i + 256],
                #     -math.sqrt(3) * stdv,
                #     math.sqrt(3) * stdv,
                # )
                # init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                # init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x):
        # (B, T, H, W, C)->(B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return F.log_softmax(x, dim=-1)


class LipNet_conv2d(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5), (2, 2), (2, 2))
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = nn.Conv2d(64, 96, (3, 3), (1, 1), (1, 1))
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))

        self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout2d = nn.Dropout2d(self.dropout_p)
        self._init()

    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

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
        B = x.size(0)
        T = x.size(1)
        # (B, T, H, W, C)->(B, T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        # (B, T, C, H, W)->(B*T, C, H, W)
        x = x.view(B * T, x.size(2), x.size(3), x.size(4))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.pool3(x)
        # (B*T, C, H, W) -> (B, T, C, H, W)
        x = x.view(B, T, x.size(1), x.size(2), x.size(3))
        # (B, T, C, H, W)->(T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return F.log_softmax(x, dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LipFormer(nn.Module):
    def __init__(self, dropout_p=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=True)
        # self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)
        self.transformer = nn.Transformer(
            d_model=96 * 3 * 6,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            dropout=dropout_p,
            activation=F.gelu,
        )
        self.emb_dim = 96 * 3 * 6
        self.output_embeddding = nn.Embedding(27 + 1, 96 * 3 * 6)

        self.FC = nn.Linear(96 * 3 * 6, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.dropout2d = nn.Dropout2d(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self.pos_enc = PositionalEncoding(d_model=self.emb_dim, dropout=dropout_p)

    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)
        return x

    def forward_train_transformer(self, x, tgt):
        # print(tgt.shape)
        tgt = tgt.permute(1, 0).contiguous()
        x = self.pos_enc(x)
        tgt = self.output_embeddding(tgt.int()) * math.sqrt(self.emb_dim)
        tgt = self.pos_enc(tgt)
        y = self.transformer(x, tgt)
        return y

    def forward_transformer(self, x):
        B = x.size(1)
        T = x.size(0)
        x = self.pos_enc(x)
        y_hat = torch.zeros((T,B,1))
        y_hat = self.output_embeddding(y_hat.int()) * math.sqrt(self.emb_dim)
        y_hat = self.pos_enc(y_hat)
        for t in range(T):
            y = self.transformer(x[t], y_hat[t])
            y_hat = torch.cat((y_hat, y), dim=0)
        return y_hat

    def forward(self, x, training=False, tgt=None):
        # (B, T, H, W, C)->(B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.forward_cnn(x)  # feature maps

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # change to squence
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)  # change to feature vectors

        # how to get tgt to have same dimention as src ???
        if training:
            assert tgt is not None
            x = self.forward_train_transformer(x, tgt)
        else:
            x = self.forward_transformer(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()  # (B, T, predictions)
        return F.log_softmax(x, dim=-1)

class LipFormerDecoder(nn.Module):
    def __init__(self, dropout_p=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # self.gru1 = nn.GRU(96 * 3 * 6, 256, 1, bidirectional=True)
        # self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)
        self.emb_dim = 96 * 3 * 6
        decoder_layer = nn.TransformerDecoderLayer(self.emb_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.output_embeddding = nn.Embedding(27 + 1, self.emb_dim)

        self.FC = nn.Linear(self.emb_dim, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.dropout2d = nn.Dropout2d(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self.pos_enc = PositionalEncoding(d_model=self.emb_dim, dropout=dropout_p)

    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)
        return x
    def forward_train_transformer(self, x, tgt):
        # print(tgt.shape)
        tgt = tgt.permute(1, 0).contiguous()
        x = self.pos_enc(x)
        tgt = self.output_embeddding(tgt.int()) * torch.sqrt(torch.tensor(self.emb_dim))
        tgt = self.pos_enc(tgt)
        y = self.transformer_decoder(tgt, x)
        return y

    def forward_transformer(self, x):
        B = x.size(1)
        T = x.size(0)
        x = self.pos_enc(x)
        y_hat = torch.zeros((T,B,1))
        y_hat = self.output_embeddding(y_hat.int()) * torch.sqrt(torch.tensor(self.emb_dim))
        y_hat = self.pos_enc(y_hat)
        for t in range(T):
            # y = self.transformer(x[t], y_hat[t])
            y = self.transformer_decoder(y_hat[t], x[t])
            y_hat = torch.cat((y_hat, y), dim=0)
        return y_hat
    
    def forward(self, x, training=False, tgt=None):
        # (B, T, H, W, C)->(B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.forward_cnn(x)  # feature maps

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # change to squence
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)  # change to feature vectors

        # how to get tgt to have same dimention as src ???
        if training:
            assert tgt is not None
            x = self.forward_train_transformer(x, tgt)
        else:
            x = self.forward_transformer(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()  # (B, T, predictions)
        return F.log_softmax(x, dim=-1)
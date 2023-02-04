import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gyprint import gyprint as g


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: torch.Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class Branch_one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Branch_one, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=out_channels),
            DropBlock(block_size=3, p=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=out_channels),
            DropBlock(block_size=3, p=0.01)
        )
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        res = self.flatten(x2)
        return res


class Branch_two(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Branch_two, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
            DropBlock(block_size=3, p=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=out_channels),
            DropBlock(block_size=3, p=0.01)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        res = self.pool(x2)
        res = torch.flatten(res, start_dim=2)
        return res


class GCNLayer1(nn.Module):
    def __init__(self, v_C, v_D, output_features, bias=False):
        super(GCNLayer1, self).__init__()
        self.v_C = v_C
        self.v_D = v_D

        self.A = nn.Parameter(torch.FloatTensor(self.v_C, self.v_C))
        self.weights = nn.Parameter(torch.FloatTensor(self.v_D, output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.LReLU = nn.LeakyReLU(negative_slope=0.01)

    def reset_parameters(self):
        # 初始化参数
        std = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-std, std)
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, v):
        v = torch.matmul(self.A, v)
        v = torch.matmul(v, self.weights)
        res = self.LReLU(v)

        if self.bias is not None:
            return res + self.bias
        return res


class GCNLayer2(nn.Module):
    def __init__(self, v_C, v_D, output_features, bias=False):
        super(GCNLayer2, self).__init__()
        self.v_C = v_C
        self.v_D = v_D
        self.E = nn.Linear(self.v_C, self.v_C)
        self.F = nn.Linear(self.v_C, self.v_C)
        self.G = nn.Linear(self.v_C, self.v_C)
        self.beita = nn.Parameter(torch.FloatTensor(self.v_C, self.v_C))
        self.weights = nn.Parameter(torch.FloatTensor(self.v_D, output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.LReLU = nn.LeakyReLU(negative_slope=0.01)

    def reset_parameters(self):
        # 初始化参数
        std = 1. / math.sqrt(self.beita.size(1))
        self.beita.data.uniform_(-std, std)
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, v_ini, v):
        E = self.E(v_ini)
        F = self.F(v_ini)
        G = self.G(v_ini) #(64, 64)
        EF = torch.matmul(E, F)
        a = torch.softmax(EF, dim=2)
        aG = torch.matmul(a, G)
        aG = torch.mul(self.beita, aG)
        A = aG + v_ini#(64, 64)
        v = torch.matmul(A, v)
        v = torch.matmul(v, self.weights)#(64, 16)
        res = self.LReLU(v)

        if self.bias is not None:
            return res + self.bias
        return res


class Branch_three(nn.Module):
    def __init__(self, in_channel, height, width, hidden_num, output_features, dropout, bias=False):
        super(Branch_three, self).__init__()
        self.D = in_channel
        self.HxW = height*width
        self.C = self.D
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.encoder_M = nn.Parameter(torch.FloatTensor(self.C, self.HxW))
        self.gcn1 = GCNLayer1(v_C=self.C, v_D=self.D, output_features=hidden_num, bias=bias)
        self.gcn2 = GCNLayer2(v_C=self.C, v_D=hidden_num, output_features=output_features, bias=bias)
        self.decoder_M = nn.Parameter(torch.FloatTensor(self.HxW, self.C))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化参数
        std = 1. / math.sqrt(self.encoder_M.size(1))
        self.encoder_M.data.uniform_(-std, std)
        std = 1. / math.sqrt(self.decoder_M.size(1))
        self.decoder_M.data.uniform_(-std, std)

    def forward(self, x):
        x_flatten = self.flatten(x)#(361, 64)
        v = torch.matmul(self.encoder_M, x_flatten)#(64, 64)
        v_layer1 = self.gcn1(v)#(64, 64)
        v_layer1 = F.dropout(v_layer1, self.dropout, training=self.training)
        v_layer2 = self.gcn2(v, v_layer1)#(64, 16)
        res = torch.matmul(self.decoder_M, v_layer2)#(361, 64)

        return res


class FDGC(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_num: int):
        super(FDGC, self).__init__()
        self.class_num = class_num  # 类别数
        self.channel = channel
        self.height = height
        self.width = width
        self.n_brach1_out = 64
        self.n_brach2_out = 128
        self.n_brach3_out = self.class_num
        self.cat_res_n = self.n_brach1_out + self.n_brach2_out + self.n_brach3_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64)
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.branch1 = Branch_one(in_channels=64, out_channels=self.n_brach1_out)
        self.branch2 = Branch_two(in_channels=64, out_channels=self.n_brach2_out)
        self.branch3 = Branch_three(in_channel=64, height=self.height,
                                    width=self.width, hidden_num=64,
                                    output_features=self.n_brach3_out, dropout=0.5)

        self.fc1 = nn.Linear(self.cat_res_n, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(1024, 256)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        #self.bn2 = nn.BatchNorm1d(256)

        self.softmax_linear = nn.Sequential(
            nn.Linear(256, self.class_num),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor):
        # 维度交换(b, c, h, w)
        # 输入为(128, 19, 19, 3)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_pool = self.pool(x2)
        #g.print_shape(x_pool)

        x_branch1 = self.branch1(x_pool)
        x_branch1 = x_branch1.permute(0, 2, 1)
        #g.print_shape(x_branch1)

        x_branch2 = self.branch2(x_pool)
        #g.print_shape(x_branch2)
        x_branch2 = x_branch2.repeat((1, 1, self.height*self.width))
        x_branch2 = x_branch2.permute(0, 2, 1)
        #g.print_shape(x_branch2)

        x2 = x2.permute(0, 2, 3, 1)
        x_branch3 = self.branch3(x2)
        #g.print_shape(x_branch3)

        x_concat = torch.cat((x_branch1, x_branch2, x_branch3), dim=2)
        #g.print_shape(x_concat)

        x_fc1 = self.fc1(x_concat)
        x_fc1 = self.act1(x_fc1)
        x_fc2 = self.fc2(x_fc1)
        x_fc2 = self.act2(x_fc2)
        res = self.softmax_linear(x_fc2)
        #g.print_shape(res)

        return res


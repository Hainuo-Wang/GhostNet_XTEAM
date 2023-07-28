import math

import torch
from torch import nn
from torchstat import stat


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # 向上取整
        new_channels = init_channels * (ratio - 1)
        # 常规普通卷积，通道浓缩
        self.ordinary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # 廉价操作，逐层卷积
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.ordinary_conv(x)
        x2 = self.cheap_conv(x1)
        # 堆叠
        out = torch.cat([x1, x2], dim=1)
        # 一般情况下特征图输出为n，不用切通道
        return out[:, :self.oup, :, :]


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            GhostModule(inp=3, oup=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=64, oup=64, kernel_size=3, stride=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=64, oup=128, kernel_size=3, stride=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=128, oup=128, kernel_size=3, stride=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            GhostModule(inp=128, oup=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            GhostModule(inp=256, oup=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            GhostModule(inp=256, oup=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=256, oup=512, kernel_size=3, stride=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=512, oup=512, kernel_size=3, stride=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=512, oup=512, kernel_size=3, stride=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=512, oup=512, kernel_size=3, stride=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=512, oup=512, kernel_size=3, stride=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            GhostModule(inp=512, oup=512, kernel_size=3, stride=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(64, 3, 32, 32)
    vgg16 = Vgg16_net()
    stat(vgg16, (3, 32, 32))
    opt = vgg16(x)
    print(opt.shape)

vgg16(x)
    # print(opt.shape)
    model = Vgg16_net()
    stat(model, (3,224,224))
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))


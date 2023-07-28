import math
import torch
import torch.nn.functional as F
from thop import profile
from torch import nn


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class Channel_Max_Pooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Channel_Max_Pooling, self).__init__()
        self.max_pooling = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        print('Input_Shape:', x.shape)  # (batch_size, chs, h, w)
        x = x.transpose(1, 3)  # (batch_size, w, h, chs)
        print('Transpose_Shape:', x.shape)
        x = self.max_pooling(x)
        print('Transpose_MaxPooling_Shape:', x.shape)
        out = x.transpose(1, 3)
        print('Output_Shape:', out.shape)
        return out


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


def _make_divisible(v, divisor, min_value=None):  # 确保所有层的通道数可被8整除，TF库内的函数，提高计算效率
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下舍入不超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GhostBottleNeck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, att_ratio=0.):
        super(GhostBottleNeck, self).__init__()
        # 开头的GhostModule有激活函数ReLU
        self.GhostModule_Start = GhostModule(in_chs, mid_chs, relu=True)
        self.channel_max_pooling = Channel_Max_Pooling((1, 2), (1, 1))
        # All the Ghost bottlenecks are applied with stride=1
        # except that the last one in each stage is with stride=2.
        self.stride = stride
        if self.stride > 1:
            # 使用深度可分离卷积对输入特征图进行高和宽的压缩
            self.depth_wise = nn.Sequential(
                nn.Conv2d(mid_chs - 1, mid_chs - 1, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2,
                          groups=mid_chs - 1, bias=False),
                nn.BatchNorm2d(mid_chs - 1)
            )
        # 某些层中有SE，具体在cfg中定义
        has_att = att_ratio is not None and att_ratio > 0.
        if has_att:
            # self.se = SE(mid_chs - 1, se_ratio=se_ratio)
            self.att = CoordAtt(mid_chs - 1, mid_chs - 1)
        else:
            self.att = None
        # 结尾的GhostModule没有激活函数ReLU
        self.GhostModule_End = GhostModule(mid_chs - 1, out_chs, relu=False)
        # 残差结构：捷径分支，stride=2的时候有，
        # 利用深度可分离卷积和1x1卷积调整通道数，保证主干部分和残差边部分能够进行相加
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        x_shortcut = x
        # 开头的ghost
        x = self.GhostModule_Start(x)
        x = self.channel_max_pooling(x)
        # 深度可分离卷积
        if self.stride > 1:
            x = self.depth_wise(x)
        # SE注意力机制
        if self.att is not None:
            x = self.att(x)
        # 结尾的ghost
        x = self.GhostModule_End(x)
        # 捷径分支
        x += self.shortcut(x_shortcut)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        out_chs = _make_divisible(16 * width, 4)  # 16层GhostBottleNeck
        # 第一层卷积
        self.conv_start = nn.Sequential(
            nn.Conv2d(3, out_chs, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )
        # 开始创建blocks
        self.cfgs = cfgs
        self.dropout = dropout
        in_chs = out_chs
        stages = []
        # 从cfg读取每一层的参数
        for cfg in self.cfgs:
            layers = []
            for k, exp, c, att_ratio, s in cfg:
                out_chs = _make_divisible(c * width, 4)
                hid_chs = _make_divisible(exp * width, 4)
                layers.append(GhostBottleNeck(
                    in_chs, hid_chs, out_chs, k, s, att_ratio=att_ratio
                ))
                in_chs = out_chs
            stages.append(nn.Sequential(*layers))
        out_chs = _make_divisible(exp * width, 4)
        # 添加一层卷积
        self.ConvBnAct = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )
        stages.append(self.ConvBnAct)
        # blocks创建完成
        self.blocks = nn.Sequential(*stages)
        # 全局池化
        self.Avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最后一层卷积
        in_chs = out_chs
        out_chs = 1280
        self.conv_end = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        # 分类层
        self.classifier = nn.Linear(out_chs, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv_start(x)
        x = self.blocks(x)
        x = self.Avg_pool(x)
        x = self.conv_end(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 1]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 1]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 1]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == '__main__':
    model = ghostnet(num_classes=10)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    # model.eval()
    # print(model)
    # input = torch.randn(64, 3, 32, 32)
    # y = model(input)
    # print(y.size())

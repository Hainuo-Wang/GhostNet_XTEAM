# import math
#
# import torch
# from torch.optim import lr_scheduler
# from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR
# import torch.nn as nn
# from torchvision.models import resnet18
# import matplotlib.pyplot as plt
#
# model = resnet18(pretrained=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# # lf = lambda x: ((1 + math.cos(x * math.pi / 300)) / 2) * (1 - 0.0001) + 0.0001  # cosine
# # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
# mode = 'cosineAnnWarm'
# # mode = 'cosineAnn'
# if mode == 'cosineAnn':
#     scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001)
# elif mode == 'cosineAnnWarm':
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
# plt.figure()
# max_epoch = 300
# iters = 391
# cur_lr_list = []
# for epoch in range(max_epoch):
#     for batch in range(iters):
#         optimizer.step()
#         scheduler.step()
#     cur_lr = optimizer.param_groups[-1]['lr']
#     cur_lr_list.append(cur_lr)
#     print('Cur lr:', cur_lr)
# x_list = list(range(len(cur_lr_list)))
# plt.plot(x_list, cur_lr_list)
# plt.show()
import torch
from backbone.GhostNet import ghostnet
backbone = ghostnet()
backbone.load_state_dict(torch.load("GhostNet.pth", map_location="cpu"))
print(backbone)

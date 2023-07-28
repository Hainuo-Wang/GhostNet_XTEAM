# import os
# import shutil
#
# # 指定train和val文件夹路径
# train_folder = "ImageNet100/train"
# val_folder = "ImageNet100/val"
#
# # 获取val文件夹中的子文件夹列表
# val_subfolders = [f.path for f in os.scandir(val_folder) if f.is_dir()]
#
# # 遍历每个子文件夹
# for subfolder in val_subfolders:
#     # 提取子文件夹名称
#     folder_name = os.path.basename(subfolder)
#     # 构建对应的train子文件夹路径
#     train_subfolder = os.path.join(train_folder, folder_name)
#
#     # 获取val子文件夹中的文件列表
#     files = os.listdir(subfolder)
#     # 遍历每个文件，复制到对应的train子文件夹中
#     for file in files:
#         file_path = os.path.join(subfolder, file)
#         shutil.copy(file_path, train_subfolder)
#
# print("复制完成！")
import torch
import torch.nn as nn


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


cmp = Channel_Max_Pooling((1, 2), (1, 1))
tensor = torch.randn((64, 3, 32, 32))
out = cmp(tensor)
print(tensor)
print(out.shape)

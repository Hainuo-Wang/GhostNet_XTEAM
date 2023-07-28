import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class MyDataSetRGB(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        img = img.convert("RGB")
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class MyDataSetL(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


# def evaluteTop1(model, loader):
#     model.eval()
#     correct = 0
#     total = len(loader.dataset)
#     loader = tqdm(loader, desc="TOP1 ", file=sys.stdout)
#     for x, y in loader:
#         x, y = x.to(args.device), y.to(args.device)
#         with torch.no_grad():
#             logits = model(x)
#             pred = logits.argmax(dim=1)
#             correct += torch.eq(pred, y).sum().float().item()
#     print("TOP1: {} %".format(correct / total * 100))
#     return correct / total


# def evaluteTop5(model, loader):
#     model.eval()
#     correct = 0
#     total = len(loader.dataset)
#     loader = tqdm(loader, desc="TOP5 ", file=sys.stdout)
#     for x, y in loader:
#         x, y = x.to(args.device), y.to(args.device)
#         with torch.no_grad():
#             logits = model(x)
#             maxk = max((1, 5))
#             y_resize = y.view(-1, 1)
#             _, pred = logits.topk(maxk, 1, True, True)
#             correct += torch.eq(pred, y_resize).sum().float().item()
#     print("TOP5: {} %".format(correct / total * 100))
#     return correct / total

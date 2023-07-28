import argparse
import datetime
import math
import os
import sys
import time
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchvision import transforms
from MyDataSet import MyDataSetRGB
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from GhostNet import ghostnet
from util import read_mydata, read_split_data

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lrf', type=float, default=0.0001)
parser.add_argument('--filename', type=str, default='result/GhostNet_imagenet100_{}.txt'.format(
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
parser.add_argument('--data_path', type=str, default="data/ImageNet100/train")
parser.add_argument('--train_data_path', type=str, default="data/ImageNet100/train")
parser.add_argument('--val_data_path', type=str, default="data/ImageNet100/val")
parser.add_argument('--pre_model', type=str, default='',
                    help='initial pre_model path')
parser.add_argument('--best_acc', type=float, default=10)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
args = parser.parse_args()

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train_images_path, train_images_label = read_mydata(args.train_data_path, 'train')
# val_images_path, val_images_label = read_mydata(args.val_data_path, 'val')
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
# 实例化训练数据集
train_dataset = MyDataSetRGB(images_path=train_images_path,
                             images_class=train_images_label,
                             transform=train_transform)

# 实例化验证数据集
test_dataset = MyDataSetRGB(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           collate_fn=train_dataset.collate_fn)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          collate_fn=test_dataset.collate_fn)

model = ghostnet(num_classes=args.num_classes).to(args.device)

if args.pre_model != "":
    if os.path.exists(args.pre_model):
        ckpt = torch.load(args.pre_model, map_location=args.device)
        new_ckpt = {}
        for idx, key in enumerate(ckpt.keys()):
            # 判断预训练模型的参数和导入模型的参数shape是否相同
            if list(ckpt.values())[idx].shape == list(model.state_dict().values())[idx].shape:
                print('===>{} matches {} successfully'.format(key, list(model.state_dict().keys())[idx]))
                # 如果相同，就以导入模型的key和预训练模型的value作为新的OrderedDict元素
                new_ckpt[list(model.state_dict().keys())[idx]] = list(ckpt.values())[idx]
            else:  # 如果shape不相同，就以导入模型的key和初始化value作为新的OrderedDict元素
                print('{} does not match {} !!!'.format(key, list(model.state_dict().keys())[idx]))
                new_ckpt[list(model.state_dict().keys())[idx]] = list(model.state_dict().values())[idx]
        miss_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
        print('load successfully!')
        print('miss keys: ', miss_keys)  # 在cnn后面又加了一层fc
        print('unexpected keys: ', unexpected_keys)

        print(model.load_state_dict(new_ckpt, strict=False))
    else:
        raise FileNotFoundError("not found pre_model file: {}".format(args.pre_model))

criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1E-4)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4E-5)
lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

if os.path.exists("../weights") is False:
    os.makedirs("../weights")
best_acc = args.best_acc


def train(epoch, train_loader):
    running_loss = 0.0
    correct = 0
    total = 0
    times = 0
    train_loader = tqdm(train_loader, desc="train", file=sys.stdout, colour="Green")
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(args.device), target.to(args.device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        times += 1
    print('epoch:%2d  loss:%.5f  train_acc:%.5f %%' % (epoch + 1, running_loss / times, 100 * correct / total))
    return running_loss / times, 100 * correct / total


def test(epoch, test_loader):
    correct = 0
    top1_correct = 0
    top5_correct = 0
    top_total = len(test_loader.dataset)
    total = 0
    global best_acc
    test_loader = tqdm(test_loader, desc="test ", file=sys.stdout, colour="red")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            top1_pred = outputs.argmax(dim=1)
            top1_correct += torch.eq(top1_pred, labels).sum().float().item()

            top5_maxk = max((1, 5))
            top5_labels = labels.view(-1, 1)
            _, top5_pred = outputs.topk(top5_maxk, 1, True, True)
            top5_correct += torch.eq(top5_pred, top5_labels).sum().float().item()

    acc = 100 * correct / total
    top1 = 100 * top1_correct / top_total
    top5 = 100 * top5_correct / top_total
    print('acc: {:.5f}%  TOP1: {:.5f}%  TOP5: {:.5f}%'.format(acc, top1, top5))
    if acc > best_acc:
        best_acc = (100 * correct / total)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    return acc, top1, top5


def record(filename, epoch, train_accuracy, val_accuracy, top1, top5, loss, lr):
    filename = filename
    data = str(epoch) + '  ' + str(train_accuracy) + '  ' + str(val_accuracy) + '  ' \
           + str(top1) + '  ' + str(top5) + '  ' + str(loss) + '  ' + str(lr)
    with open(filename, 'a') as f:
        f.write(data)
        f.write('\n')


def record_time(filename, runningtime):
    with open(filename, 'a') as f:
        f.write(runningtime)
        f.write('\n')


if __name__ == '__main__':
    start = time.perf_counter()
    filename = args.filename
    title = 'epoch' + '  ' + 'accuracy_train' + '  ' + 'accuracy_val' + '  ' + 'loss' + '  ' + 'learnning_rate'
    with open(filename, 'a') as f:
        f.write(title)
        f.write('\n')
    total_accuracy = []
    for epoch in range(args.epochs):
        loss, train_accuracy = train(epoch, train_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        val_accuracy, TOP1, TOP5 = test(epoch, test_loader)
        record(filename, epoch, train_accuracy, val_accuracy, TOP1, TOP5, loss, lr)
    end = time.perf_counter()
    running_time = 'runningtime:' + '  ' + str((end - start) // 60) + 'min' + '  ' + str((end - start) % 60) + 's'
    record_time(filename, running_time)

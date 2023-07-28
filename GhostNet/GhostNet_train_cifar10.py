import argparse
import os
import sys
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from GhostNet.model.GhostNet_CIFAR import ghostnet

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lrf', type=float, default=0.0001)
parser.add_argument('--filename', type=str, default='result/GhostNet_CIFAR10.txt')
parser.add_argument('--data_path', type=str, default="data/CIFAR10")
parser.add_argument('--pre_model', type=str, default='',
                    help='initial pre_model path')
parser.add_argument('--best_acc', type=float, default=90)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
args = parser.parse_args()

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447],
                         std=[0.247, 0.243, 0.262])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447],
                         std=[0.247, 0.243, 0.262])
])

train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

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
# scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200, 250], gamma=0.1)
# pg = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
# lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
# optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1E-4)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

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
    print('epoch:%2d  loss:%.5f  train_acc:%.5f' % (epoch + 1, running_loss / times, 100 * correct / total))
    return running_loss / times, 100 * correct / total


def test(epoch, test_loader):
    correct = 0
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
    print('Accuracy on test set:%.5f %%' % (100 * correct / total))
    if (100 * correct / total) > best_acc:
        best_acc = (100 * correct / total)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    return 100 * correct / total


def record(filename, epoch, train_accuracy, val_accuracy, loss, lr):
    filename = filename
    data = str(epoch) + '  ' + str(train_accuracy) + '  ' + str(val_accuracy) + '  ' + str(loss) + '  ' + str(lr)
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
        val_accuracy = test(epoch, test_loader)
        total_accuracy.append(val_accuracy)
        record(filename, epoch, train_accuracy, val_accuracy, loss, lr)
    end = time.perf_counter()
    running_time = 'runningtime:' + '  ' + str((end - start) // 60) + 'min' + '  ' + str((end - start) % 60) + 's'
    record_time(filename, running_time)

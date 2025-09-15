from random import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import csv
import time
from models.MobileNetV2 import MobileNetV2
from models.ShuffleNetV2 import ShuffleNetV2
from util import test
import random
import torch.nn.functional as F
from torchvision import transforms, datasets

seed = 42


def parms(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 4  # 如果是浮点数就是4
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para / 1e6) + 'M')
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


# 设置种子（例如42）
if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    # model = MobileNetV2(6,100)
    model = ShuffleNetV2()
    input = torch.randn(1, 3, 224, 224)
    model.eval()
    from fvcore.nn import FlopCountAnalysis
    x = FlopCountAnalysis(model, input)
    print(x.total())
    parms(model)
    path_csv = './ImageNet-100-S2.csv'
    path_pkl = './ImageNet-100-S2.pkl'
    batchSize_train = 128
    batchSize_val = 128
    # write header
    # ImageNet-100 数据增强和标准化
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整到 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准化
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),  # 先调整到 256x256
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准化
    ])

    # 加载 ImageNet-100（假设数据路径为 "../.temp/imagenet100"）
    trainset = datasets.ImageFolder(
        root="/home/ac/data/wfz/imagenet100/train",  # train 文件夹（包含 100 个子文件夹，每个类一个）
        transform=transform_train
    )
    testset = datasets.ImageFolder(
        root="/home/ac/data/wfz/imagenet100/val",  # val 文件夹（同样按类别存放）
        transform=transform_test
    )
    # DataLoader（调整 batch_size 以适应 GPU 显存）
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batchSize_train,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batchSize_val,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    with open(path_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])
    model = model.to(device)
    # criterion = LabelSmoothingCrossEntropy()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,  # 初始学习率
        momentum=0.9,  # Nesterov 动量
        weight_decay=4e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    max_val_acc = 0
    start_time = time.time()
    for epoch in range(150):
        epoch_start_time = time.time()
        correct, total = 0, 0
        train_loss, counter = 0, 0
        model.train()
        for data in trainloader:
            # 获取输入数据
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练损失和正确率
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        print("\n")
        # 计算训练集的平均损失和正确率
        train_loss /= len(trainloader)
        train_acc = correct / total
        # 在测试集上评估模型
        val_loss, val_acc = test(model, testloader, criterion)

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])
        if val_acc > max_val_acc:
            torch.save(model.state_dict(), path_pkl)
            max_val_acc = val_acc
        # 输出当前 epoch 的时间
        # 输出当前 epoch 的结果
        print(f'Epoch {epoch + 1}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}',
              f'Max Val Acc: {max_val_acc}')
        print(f'Epoch time: {(time.time() - epoch_start_time) / 60:.4f} min')

    # 输出总训练时间
    total_time = (time.time() - start_time) / 3600
    print(f'Total training time: {total_time:.1f} h')

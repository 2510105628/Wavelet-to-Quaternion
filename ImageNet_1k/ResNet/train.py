import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from thop import profile

from networks.imagenet import create_net
from util import AverageMeter, ProgressMeter, accuracy
from checkpoint import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training (DDP + torchrun only)')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
# 注意：这里的 batch-size 定义为【每张 GPU 的 batch】
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--pretrained', action='store_true')

# DDP 相关（仅保留 backend / dist-url=env://）
parser.add_argument('--dist-url', default='env://', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str)

# 其他功能
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--ckpt', default='./ckpts/')
parser.add_argument('--attention_type', type=str, default='none')
parser.add_argument('--attention_param', type=float, default=4)
parser.add_argument('--log_freq', type=int, default=500)
parser.add_argument('--cos_lr', action='store_true')
parser.add_argument('--save_weights', default=None, type=str)

best_acc1 = 0


def is_main_process(args):
    return args.rank == 0


def setup_ddp_from_torchrun(args):
    """从 torchrun 注入的环境变量中初始化 DDP。"""
    assert 'LOCAL_RANK' in os.environ, "This script must be launched with torchrun."
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    args.distributed = True
    return torch.device(f'cuda:{args.local_rank}')


def main():
    args = parser.parse_args()

    # 目录与命名
    args.ckpt = os.path.join(
        args.ckpt,
        f"imagenet-{args.arch}" + ("" if args.attention_type.lower() == "none"
                                   else f"-{args.attention_type}-param{args.attention_param}") +
        f"-seed{args.seed}"
    )
    os.makedirs(args.ckpt, exist_ok=True)

    # 随机数种子（可复现）
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('Deterministic CUDNN may slow down training.')

    # 仅支持 torchrun 启动
    device = setup_ddp_from_torchrun(args)

    # 创建与放置模型
    model = create_net(args).to(args.local_rank)

    # # 只在 rank0 打印 FLOPs/Params 并记录
    # if is_main_process(args):
    #     x = torch.randn(1, 3, 224, 224, device=device)
    #     flops, params = profile(module, inputs=(x,))
    #     print(f"module [{args.arch}] - params: {params/1e6:.6f}M")
    #     print(f"module [{args.arch}] - FLOPs: {flops/1e9:.6f}G")
    #     with open(os.path.join(args.ckpt, "log.txt"), "a") as f:
    #         f.write(f"Network - {args.arch}\n")
    #         f.write(f"Attention Module - {args.attention_type}\n")
    #         f.write(f"Params - {params}\n")
    #         f.write(f"FLOPs - {flops}\n")
    #         f.write("--------------------------------------------------\n")
    # dist.barrier()
    # DDP 封装
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank]
    )
    cudnn.benchmark = True

    # 损失与优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # 恢复
    global best_acc1
    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch

    # 数据
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    # DistributedSampler（train/val 都用）
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
    )

    # 注意：这里的 batch-size 是【每 GPU 的 batch】
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 使用 DistributedSampler 时应为 False
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    # 仅导出“去并行化”权重（直接退出）
    if args.save_weights is not None:
        if is_main_process(args):
            print(f"=> saving 'deparallelized' weights [{args.save_weights}]")
            to_save = model.module if hasattr(model, "module") else model
            torch.save({'state_dict': to_save.cpu().state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)
        dist.barrier()
        dist.destroy_process_group()
        return

    # 仅评估
    if args.evaluate:
        acc1 = validate(val_loader, model, criterion, device, args)
        if is_main_process(args):
            print('----------- Acc@1 {:.3f} -----------'.format(acc1))
        dist.barrier()
        dist.destroy_process_group()
        return

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.cos_lr else None
    if scheduler is not None:
        for _ in range(args.start_epoch):
            scheduler.step()

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        if scheduler is None:
            adjust_learning_rate(optimizer, epoch, args)
            lr_show = optimizer.param_groups[0]["lr"]
        else:
            scheduler.step()
            lr_show = scheduler.get_last_lr()[0]

        if is_main_process(args):
            print(f"[{epoch:03d}] lr={lr_show:.6f}")

        train(train_loader, model, criterion, optimizer, epoch, device, args)
        acc1 = validate(val_loader, model, criterion, device, args)

        if is_main_process(args):
            print('----------- Acc@1 {:.3f} -----------'.format(acc1))
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "best_acc": max(acc1, best_acc1),
                "optimizer": optimizer.state_dict(),
            }, is_best=(acc1 > best_acc1), epoch=epoch, save_path=args.ckpt)
            best_acc1 = max(acc1, best_acc1)

        dist.barrier()  # 同步各进程，避免 rank0 太快结束

    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_process(args):
            msg = progress.get_message(i) + f"\tLr  {optimizer.param_groups[0]['lr']:.6f}"
            print(msg)


@torch.no_grad()
def validate(val_loader, model, criterion, device, args):
    """验证阶段做全进程 all-reduce，返回全数据集 Acc@1。"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_process(args):
            print(f"Test: [{i}/{len(val_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.4e} ({losses.avg:.4e})\t"
                  f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                  f"Acc@5 {top5.val:.2f} ({top5.avg:.2f})")

    # 全进程聚合（样本加权平均）
    for meter in (losses, top1, top5):
        sum_tensor = torch.tensor([meter.sum], device=device, dtype=torch.float64)
        cnt_tensor = torch.tensor([meter.count], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
        meter.sum = sum_tensor.item()
        meter.count = int(cnt_tensor.item())
        meter.avg = meter.sum / max(1, meter.count)

    return float(top1.avg)


def adjust_learning_rate(optimizer, epoch, args):
    """Step LR: 每 30 个 epoch 衰减 10 倍（当未使用 Cosine 时）"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

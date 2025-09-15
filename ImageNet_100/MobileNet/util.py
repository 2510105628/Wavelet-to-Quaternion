import torch


def test(model, testloader, criterion):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0
    total_loss = 0.0  # 修复问题2：重命名变量

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 统计准确率
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()

            # 统计损失（按总损失累加）
            batch_loss = criterion(outputs, labels).item()
            total_loss += batch_loss * batch_size  # 修复问题2：加权总损失

    val_loss = total_loss / total  # 全体样本平均损失
    val_acc = correct / total
    return val_loss, val_acc
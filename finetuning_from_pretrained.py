import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.models import resnet18  # 导入torchvision库中的resnet18
from torch.utils.tensorboard import SummaryWriter
from load_cifar100 import load_cifar100_data
from tqdm import tqdm


def linear_evaluation(epochs, batch_size, learning_rate):
    train_loader, test_loader = load_cifar100_data(batch_size)
    writer = SummaryWriter('runs/pretrained_imagenet_linear_classification')

    # 加载在ImageNet上预训练的ResNet18
    base_model = resnet18(pretrained=True).cuda()  # 设置pretrained=True来加载预训练权重
    base_model.fc = nn.Linear(base_model.fc.in_features, 100).cuda()  # 替换最后一个fc层以匹配CIFAR-100的100个类

    # 冻结除最后一层外的所有层
    for param in base_model.parameters():
        param.requires_grad = False

    # 只训练最后一层
    base_model.fc.weight.requires_grad = True
    base_model.fc.bias.requires_grad = True

    optimizer = Adam(base_model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练线性分类器
    for epoch in tqdm(range(epochs)):
        base_model.train()
        total_loss, total_correct, total_images, total_batches = 0, 0, 0, 0
        for images, labels in train_loader:
            total_batches += 1
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

        train_accuracy = total_correct / total_images
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        print(f'Epoch {epoch}: Loss {total_loss / total_batches}, Accuracy {total_correct / total_images}')

        # 测试
        base_model.eval()
        total_correct, total_images = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = base_model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
        test_accuracy = total_correct / total_images
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy}')

        if epoch == epochs - 1:
            torch.save(base_model, f'../final_models/pretrained_imagenet_model_epoch_{epoch}.pth')
            torch.save(base_model.state_dict(), f'../final_models/pretrained_imagenet_state_dict_epoch_{epoch}.pth')

    writer.close()

if __name__ == "__main__":
    linear_evaluation(50, 128, 0.01)

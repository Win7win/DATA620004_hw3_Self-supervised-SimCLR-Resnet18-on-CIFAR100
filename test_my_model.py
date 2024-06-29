import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from resnet_18 import get_ResNet18
from torch.utils.tensorboard import SummaryWriter
from load_cifar100 import load_cifar100_data
from tqdm import tqdm

def test_model(pretrained_path, batch_size, out_dimen, learning_rate):
    train_loader, test_loader = load_cifar100_data(batch_size)
    writer = SummaryWriter(f'runs/cifar100_linear_classification_test')

    # 加载预训练权重
    base_model = get_ResNet18(out_dimension=out_dimen).cuda()
    base_model.load_state_dict(torch.load(pretrained_path))

    # 冻结特征提取层
    for param in base_model.feature_extractor.parameters():
        param.requires_grad = False

    # 移除投影头，添加分类层
    num_features = 512  # 假定特征提取层的输出为512维
    base_model.projection_head = nn.Linear(num_features, 100).cuda()  # CIFAR-100分类层
    model = base_model

    # 检查冻结状态
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(f"{name} is not frozen")
        else:
            print(f"{name} is frozen")

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 运行测试
    model.eval()
    total_correct, total_images = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)
    test_accuracy = total_correct / total_images
    writer.add_scalar('Accuracy/test', test_accuracy)
    print(f'Test Accuracy: {test_accuracy}')

    writer.close()

if __name__ == "__main__":
    pretrained_path = '../final_models/stl-10-model_state_dict_self_supervised_cifar100_epoch_49_0.01_128_0.5.pth' 
    batch_size = 128
    out_dimen = 100  
    learning_rate = 0.01

    test_model(pretrained_path, batch_size, out_dimen, learning_rate)

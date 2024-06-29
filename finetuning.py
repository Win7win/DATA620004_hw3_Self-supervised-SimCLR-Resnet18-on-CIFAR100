import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from resnet_18 import get_ResNet18
from torch.utils.tensorboard import SummaryWriter
from load_cifar100 import load_cifar100_data
from tqdm import tqdm


def linear_evaluation(pretrain_params, pretrained_path, epochs, batch_size, out_dimen, learning_rate):
    train_loader, test_loader = load_cifar100_data(batch_size)
    writer = SummaryWriter(f'runs/stl10-self-surpervise_linear_classification_cifar100/{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}')

    # 加入自监督预训练权重
    base_model = get_ResNet18(out_dimension=out_dimen).cuda()
    base_model.load_state_dict(torch.load(pretrained_path))


    # 冻结特征提取层
    for param in base_model.feature_extractor.parameters():
        param.requires_grad = False

    # 移除投影头，添加分类层
    num_features = 512  # 假定特征提取层的输出为512维
    base_model.projection_head = nn.Linear(num_features, 100).cuda()  # CIFAR-100分类层
    model = base_model
    # 再次检查冻结状态
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(f"{name} is not frozen")
        else:
            print(f"{name} is frozen")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练线性分类器
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss, total_correct, total_images, total_batches = 0, 0, 0, 0
        for images, labels in train_loader:
            total_batches += 1
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
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
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy}')
        if epoch == epochs - 1:
            torch.save(model, f'../final_models/stl-10-full_model_self_supervised_cifar100_epoch_{epoch}_{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}.pth') 
            torch.save(model.state_dict(), f'../final_models/stl-10-model_state_dict_self_supervised_cifar100_epoch_{epoch}_{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}.pth')  
    with open("stl10_records.txt", "a+", encoding="utf-8") as f:
        f.write(str(pretrain_params) + str(test_accuracy) + "\n")
    writer.close()

if __name__ == "__main__":
    for lr in [0.01, 0.001, 0.005]:
        for batch in [64, 128]:
            for gam in [0.5, 0.7]:
                if lr == 0.001 and batch == 128 and gam==0.7:
                    continue
                ori_params = [lr, batch, gam]
                linear_evaluation(ori_params, f'/root/model_saved/neural_network_hw3/task1/pretrained_models/stl_10/resnet18_epoch_50_{lr}_{batch}_{gam}.pth', 50, 128, 128, 0.01)


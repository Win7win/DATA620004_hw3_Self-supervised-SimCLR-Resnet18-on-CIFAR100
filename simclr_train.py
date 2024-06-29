import torch
import torch.optim as optim
from resnet_18 import get_ResNet18
from loss_function import nt_xent_loss
from load_stl10 import load_stl10_data
from load_cifar100 import load_cifar100_data
from load_cifar10 import load_cifar10_data
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def pre_train_simclr(epochs, learning_rate, batchsize=128, out_dimen=128, save_steps=5, step_size=20, gamma=0.5, dataset="stl_10"):
    writer = SummaryWriter(f'runs/{dataset}/self-supervised/resnet_18_{learning_rate}_{batchsize}_{gamma}')

    # 载入数据和模型
    if dataset == "stl_10":
        data = load_stl10_data(batchsize)
    elif dataset == "cifar_10":
        data, _ = load_cifar10_data(batchsize)
    else:
        print("not predefined dataset")
        return 
    model = get_ResNet18(out_dimension=out_dimen).cuda()
    # 优化器和学习率递减
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("开始训练")
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in data:
            images, _ = batch
            images = images.cuda()
            optimizer.zero_grad()
            z_i = model(images)
            z_j = model(images)

            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f'Epoch {epoch}, Loss: {average_loss}')
        writer.add_scalar('Loss/train', average_loss, epoch)
        
        if (epoch + 1) % save_steps == 0:
            torch.save(model.state_dict(), f'../pretrained_models/{dataset}/resnet18_epoch_{epoch+1}_{learning_rate}_{batchsize}_{gamma}.pth')
        
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    for lr in [0.01, 0.001, 0.005]:
        for batch in [64, 128]:
            for gam in [0.5, 0.7]:
                if lr == 0.001 and batch == 128 and gam==0.7:
                    continue
                pre_train_simclr(epochs=50, learning_rate=lr, batchsize=batch, out_dimen=128, save_steps=10, step_size=10, gamma=gam, dataset="stl_10")

# Self-supervised-SimCLR-Resnet18-on-CIFAR100

复旦大学 DATA620004 神经网络和深度学习 期末作业

## 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现

### 基本要求：
1. **实现任一自监督学习算法** 并使用该算法在自选的数据集上训练 ResNet-18，随后在 CIFAR-100 数据集中使用 Linear Classification Protocol 对其性能进行评测。
2. **将上述结果与在 ImageNet 数据集上采用监督学习训练得到的表征在相同的协议下进行对比**，并比较二者相对于在 CIFAR-100 数据集上从零开始以监督学习方式进行训练所带来的提升。
3. **尝试不同的超参数组合**，探索自监督预训练数据集规模对性能的影响。

### 准备
请确保安装以下依赖：
- torch
- torchvision
- tensorboard
- tqdm

### 训练
1. **自监督预训练**
   ```sh
   python simclr_train.py
   ```
2. **在 CIFAR-100 上执行 Linear Classification Protocol**
   ```sh
   python finetuning.py
   ```
3. **在 ImageNet 数据集上采用监督学习训练得到的表征在相同的协议下进行训练，以实现对比**
   ```sh
   python finetuning_from_pretrained.py
   ```
4. **在 CIFAR-100 数据集上从零开始以监督学习方式进行训练**
   ```sh
   python train_from_scratch.py
   ```

### 测试
运行以下脚本对模型进行测试：
```sh
python test_my_model.py
```

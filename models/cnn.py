from torch import nn
import torch.nn.functional as F
import torch
"""
Small CNN Architectures taken from
https://github.com/JianXu95/FedPAC/blob/main/models/cnn.py
"""

class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CIFARNet, self).__init__()
        self.input_shape = (in_channels, 32, 32)  # CIFAR默认尺寸
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.flat_size = 64 * 3 * 3
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes
        # Define keys for the classifier layer
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)
        if return_feat:
            return x, out
        return out

class EMNISTNet(nn.Module):
    def __init__(self, num_classes=62, in_channels=1):
        super(EMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.flat_size = 32 * 5 * 5
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.D = 128
        self.cls = num_classes

    def forward(self, x, return_feat=False):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)
        if return_feat:
            return x, out
        return out


class ImageNet(nn.Module):
    # 添加 input_size 参数，用于指定输入图片的尺寸
    def __init__(self, num_classes=10, in_channels=3, input_size=64): # Default to 64 for Tiny-ImageNet
        super(ImageNet, self).__init__()
        # 更新 input_shape 以反映实际输入的尺寸
        self.input_shape = (in_channels, input_size, input_size)

        # 定义卷积和池化层 (这部分网络结构不变)
        self.conv1 = nn.Conv2d(in_channels, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # --- 动态计算 self.flat_size ---
        # 创建一个只包含特征提取部分 (conv + pool) 的临时 Sequential 模型
        # 需要确保这里的层和顺序与 forward 方法中的一致，包括激活函数
        self.features_temp = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(), # 对应 forward 中的 F.leaky_relu
            self.pool,
            self.conv2,
            nn.LeakyReLU(),
            self.pool,
            self.conv3,
            nn.LeakyReLU(),
            self.pool
        )

        # 创建一个虚拟的输入张量，用于计算展平后的尺寸
        # 尺寸为 [批量大小=1, 通道数, 高, 宽]
        dummy_input = torch.randn(1, in_channels, input_size, input_size)

        # 将虚拟输入通过特征提取层
        dummy_output = self.features_temp(dummy_input)

        # 计算展平后的元素个数 (排除批量大小维度)
        self.flat_size = torch.flatten(dummy_output, 1).size(1)

        # --- 使用计算出的 self.flat_size 定义全连接层 ---
        self.linear = nn.Linear(self.flat_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.D = 128
        self.cls = num_classes
        # Define keys for the classifier layer
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x, return_feat=False):
        # 直接使用 self.features_temp 来处理特征提取部分，更简洁
        # 或者保持原样使用单独的层调用，但确保与 __init__ 中的顺序一致
        # 这里保持原样，但逻辑上等同于 self.features_temp(x)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))

        # 使用计算出的 self.flat_size 进行展平操作
        x = x.view(-1, self.flat_size)

        # 传递给全连接层
        x = F.leaky_relu(self.linear(x))
        out = self.fc(x)

        if return_feat:
            return x, out
        return out

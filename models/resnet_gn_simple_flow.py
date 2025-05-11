import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from .group_normalization import GroupNorm2d
import math

class RectifiedFlowModule(nn.Module):
    def __init__(self, latent_dim, num_steps=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        # 连续时间编码维度
        self.time_embed_dim = latent_dim
        
        # 初始化调整
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, 4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),  # 添加层归一化
            nn.GELU(),
            nn.Linear(4 * latent_dim, 4 * latent_dim),
            nn.LayerNorm(4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim)
        )
        self.attention = nn.Sequential(  # 新增注意力门控
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        # 初始化最后一层为接近零
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

    def forward(self, z):
        # 添加输入归一化
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        
        batch_size = z.size(0)
        t = torch.rand(batch_size, device=z.device) * self.num_steps
        
        time_emb = self.sinusoidal_embedding(t, self.time_embed_dim)
        h = torch.cat([z, time_emb], dim=1)
        
        delta = self.mlp(h)
        
        # 限制修正幅度
        delta = torch.tanh(delta) * 0.1  # 输出限制在[-0.1, 0.1]
        
        t_factor = t / self.num_steps
        # 在残差连接前加入门控机制
        gate = self.attention(z)
        return z + gate * delta * t_factor.unsqueeze(-1)
        # return  z + delta * t_factor.unsqueeze(-1)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

norm2d = lambda planes: GroupNorm2d(planes, num_groups=32, affine=True, track_running_stats=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
        has_bn = True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = norm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn3 = norm_layer(planes)
        else:
            self.bn3 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        has_bn = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = norm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if has_bn:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if has_bn:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion)
        if has_bn:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        features: List[int] = [64, 128, 256, 512],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
        has_bn = True,
        bn_block_num = 4, 
        in_channels=3
    ) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = norm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0)))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                       dilate=replace_stride_with_dilation[num-1], 
                                       has_bn=has_bn and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        self.D = features[len(layers)-1] * block.expansion
        self.feature_refiner = RectifiedFlowModule(latent_dim=self.D)  # 新增特征整流模块

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_bn=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_bn))

        return layers

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        feat = self.avgpool(x)
        # 新增特征分布整流
        refined_feat = self.feature_refiner(feat)  # 将特征推向高斯流形
        out = self.fc(refined_feat)
        if return_feat:
            return refined_feat, out
        else:
            return out

def resnet152(**kwargs: Any) -> ResNet: 
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnet101(**kwargs: Any) -> ResNet: 
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet50(**kwargs: Any) -> ResNet: 
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet34(**kwargs: Any) -> ResNet: 
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet18(**kwargs: Any) -> ResNet: # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet10(**kwargs: Any) -> ResNet: # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)

def resnet8(**kwargs: Any) -> ResNet: # 8 = 2 + 2 * (1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)

def resnet6(**kwargs: Any) -> ResNet: # 6 = 2 + 2 * (1 + 1)
    return ResNet(BasicBlock, [1, 1], **kwargs)

def resnet4(**kwargs: Any) -> ResNet: # 4 = 2 + 2 * (1)
    return ResNet(BasicBlock, [1], **kwargs)

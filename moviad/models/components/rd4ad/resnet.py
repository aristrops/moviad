import torch
from torch import Tensor
import torch.nn as nn
import timm
import math
import torch.nn.functional as F
from functools import partial
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'MobileNetV2Backbone', 'BN_layer_mobilenet', 'mobilenet_v2_rd4ad']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
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
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        # Bottleneck, [3, 4, 6, 3],

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        feature_d = self.layer4(feature_c)


        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    #_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
    #               pretrained, progress, **kwargs)
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

class AttnBasicBlock(nn.Module):
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
        attention: bool = True,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        #print("Attention:", self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        #self.cbam = GLEAM(planes, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        #if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AttnBottleneck(nn.Module):
    
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
        attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        #print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #self.cbam = GLEAM([int(planes * self.expansion/4),
        #                   int(planes * self.expansion//2),
        #                   planes * self.expansion], 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        #if self.attention:
        #    x = self.cbam(x)
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

class BN_layer(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes*3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes*3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #x = self.cbam(x)
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1,l2,x[2]],1)
        output = self.bn_layer(feature)
        #x = self.avgpool(feature_d)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(pretrained: bool = False, progress: bool = True,**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs), BN_layer(AttnBasicBlock,2,**kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBasicBlock,3,**kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBasicBlock,3,**kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


# ── MobileNetV2 backbone (encoder) ────────────────────────────────────────────
class MobileNetV2Backbone(nn.Module):
    """
    Wraps torchvision's MobileNetV2 and exposes the same three feature maps
    that ResNet exposes via layer1/layer2/layer3 (strides 4 / 8 / 16).

    MobileNetV2 feature stages tapped:
        features[0:2]  → stride 2  (entry stem, not tapped)
        features[2:4]  → stride 4   ← feature_a  (24 ch)
        features[4:7]  → stride 8   ← feature_b  (32 ch)
        features[7:14] → stride 16  ← feature_c  (96 ch)

    forward() returns [feature_a, feature_b, feature_c], matching the
    list contract of ResNet._forward_impl().
    """

    # Output channels at each tapped stage — used by BN_layer_mobilenet
    C1: int = 24  # stride 4
    C2: int = 32  # stride 8
    C3: int = 96  # stride 16

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        feats = mobilenet_v2(weights=weights).features  # nn.Sequential of 19 blocks

        self.layer0 = feats[0:2]  # stride 2  – entry stem
        self.layer1 = feats[2:4]  # stride 4  – 24 ch  → feature_a
        self.layer2 = feats[4:7]  # stride 8  – 32 ch  → feature_b
        self.layer3 = feats[7:14]  # stride 16 – 96 ch  → feature_c

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.layer0(x)
        feature_a = self.layer1(x)  # (B, 24, H/4,  W/4)
        feature_b = self.layer2(feature_a)  # (B, 32, H/8,  W/8)
        feature_c = self.layer3(feature_b)  # (B, 96, H/16, W/16)
        return [feature_a, feature_b, feature_c]


# ── BN_layer_mobilenet ────────────────────────────────────────────────────────

class BN_layer_mobilenet(nn.Module):
    """
    BN_layer adapted for a MobileNetV2 backbone.

    Mirrors BN_layer from the ResNet encoder file exactly in structure:
      - conv1 path: projects x[0] (C1=24, stride 4) down to stride 16
      - conv3 path: projects x[1] (C2=32, stride 8) down to stride 16
      - x[2]      : (C3=96, stride 16) passed through directly
      - concatenation of all three → bn_layer stack → output

    Channel constants are chosen so that the concatenated tensor
    (OUT_C * 3 = 288) feeds a bn_layer that produces BN_OUT_C=192 channels,
    which the paired MobileNetV2Decoder consumes.

    The _make_layer / forward structure is a faithful translation of BN_layer,
    with the block.expansion-based arithmetic replaced by explicit channel
    values derived from MobileNetV2Backbone.layer_channels.
    """

    # Projection target: each of the three branches is brought to OUT_C channels
    # before concatenation. OUT_C matches C3 so x[2] needs no projection conv.
    OUT_C: int = 96  # = C3
    # Output of the bn_layer residual stack (consumed by the decoder)
    BN_OUT_C: int = 192

    def __init__(
            self,
            layers: int,  # number of blocks in bn_layer
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs,  # absorbs groups/width_per_group etc.
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # AttnBasicBlock requires groups=1, base_width=64 — always use those
        self.groups = 1
        self.base_width = 64
        self.dilation = 1

        C1, C2, C3 = MobileNetV2Backbone.C1, MobileNetV2Backbone.C2, MobileNetV2Backbone.C3
        OUT_C, BN_OUT_C = self.OUT_C, self.BN_OUT_C

        self.relu = nn.ReLU(inplace=True)

        # ── conv1 path: C1 (24) at stride-4 → OUT_C (96) at stride-16 ─────
        # Two strided 3×3 convolutions, each ×2 downsample (mirrors conv1+conv2
        # in BN_layer which applies two consecutive stride-2 convolutions on x[0])
        self.conv1 = conv3x3(C1, OUT_C // 2, stride=2)  # stride 4  → 8  (24 → 48)
        self.bn1 = norm_layer(OUT_C // 2)
        self.conv2 = conv3x3(OUT_C // 2, OUT_C, stride=2)  # stride 8  → 16 (48 → 96)
        self.bn2 = norm_layer(OUT_C)

        # ── conv3 path: C2 (32) at stride-8 → OUT_C (96) at stride-16 ─────
        # One strided 3×3 convolution (mirrors conv3 in BN_layer on x[1])
        self.conv3 = conv3x3(C2, OUT_C, stride=2)  # stride 8  → 16 (32 → 96)
        self.bn3 = norm_layer(OUT_C)

        # ── x[2] path: C3=96=OUT_C at stride-16, no projection needed ─────

        # ── bn_layer: AttnBasicBlock stack on cat([l1, l2, x[2]]) ──────────
        # Input channels = OUT_C * 3 = 288  (mirrors inplanes*3 in BN_layer)
        self.inplanes = OUT_C * 3  # 288
        self.bn_layer = self._make_layer(BN_OUT_C, layers, stride=2)

        # Weight init (identical to BN_layer)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """
        Mirrors BN_layer._make_layer exactly.
        First block: inplanes (=OUT_C*3=288) → planes with optional downsample.
        Subsequent blocks: planes → planes.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # Downsample branch: uses inplanes (not inplanes*3, since we set
        # self.inplanes = OUT_C*3 before calling this, matching the first
        # block's actual input width)
        if stride != 1 or self.inplanes != planes * AttnBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * AttnBasicBlock.expansion, stride),
                norm_layer(planes * AttnBasicBlock.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(
            AttnBasicBlock(self.inplanes, planes,     stride, downsample,
                           self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * AttnBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                AttnBasicBlock(self.inplanes, planes,
                               groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: List[Tensor]) -> Tensor:
        """
        Mirrors BN_layer._forward_impl:
            l1 = project x[0] (stride 4)  to stride 16
            l2 = project x[1] (stride 8)  to stride 16
            concatenate [l1, l2, x[2]]
            pass through bn_layer
        """
        # x[0]: (B, 24, H/4,  W/4)
        l1 = self.relu(self.bn1(self.conv1(x[0])))  # → (B, 48,  H/8,  W/8)
        l1 = self.relu(self.bn2(self.conv2(l1)))  # → (B, 96,  H/16, W/16)

        # x[1]: (B, 32, H/8,  W/8)
        l2 = self.relu(self.bn3(self.conv3(x[1])))  # → (B, 96,  H/16, W/16)

        # x[2]: (B, 96, H/16, W/16) — no projection needed
        feature = torch.cat([l1, l2, x[2]], dim=1)  # → (B, 288, H/16, W/16)

        output = self.bn_layer(feature)  # → (B, 192, H/32, W/32)
        return output.contiguous()

    def forward(self, x: List[Tensor]) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2_rd4ad(pretrained: bool = False,
                       progress: bool = True,
                       **kwargs: Any):
    backbone = MobileNetV2Backbone(pretrained=pretrained)
    bn_layer = BN_layer_mobilenet(layers=2, **kwargs)
    return backbone, bn_layer


## ── DEIT small backbone ────────────────────────────────────────────────────────
class DeiTSmallBackbone(nn.Module):
    """
    Wraps a pretrained DeiT-Small ViT and exposes three pseudo-hierarchical
    feature maps that match the spatial-pyramid contract of ResNet / MobileNetV2:

        feature_a : (B, C1, H/4,  W/4)   ← shallow transformer layers
        feature_b : (B, C2, H/8,  W/8)   ← mid transformer layers
        feature_c : (B, C3, H/16, W/16)  ← deep transformer layers

    Since DeiT-Small (patch_size=16) produces a single spatial resolution of
    H/16 × W/16 from all its layers, we simulate the multi-scale pyramid by:

      1. Tapping three evenly-spaced checkpoints inside the 12 transformer
         blocks (layers 3, 7, 11 by default).
      2. Reshaping each checkpoint's patch tokens back into a 2-D grid at
         stride 16 (H/16 × W/16) and reducing the 384-d token dim to C3=192.
      3. Bilinearly upsampling the shallow/mid checkpoints to stride 8 and
         stride 4, matching the ResNet spatial contract exactly.
      4. Projecting each upsampled map to C1 / C2 / C3 channels via a
         lightweight 1×1 conv so downstream BN_layer_deit sees known widths.

    This design keeps the encoder output API identical to MobileNetV2Backbone
    and avoids any modification to the shared BN_layer / decoder code.

    Channel constants (used by BN_layer_deit):
        C1 = 96   (stride-4  map, finest)
        C2 = 192  (stride-8  map)
        C3 = 384  (stride-16 map, coarsest, = DeiT hidden dim)

    Args:
        pretrained  : load DeiT-Small ImageNet weights via timm
        img_size    : expected input spatial size (default 224); must be
                      divisible by patch_size (16)
        tap_layers  : indices (0-based) of the 12 transformer blocks to tap;
                      defaults to (3, 7, 11) for shallow / mid / deep
    """

    # Output channels of the three projected feature maps
    C1: int = 96  # stride-4
    C2: int = 192  # stride-8
    C3: int = 384  # stride-16  (= DeiT-Small hidden dim, no projection needed)

    _DEIT_DIM: int = 384  # DeiT-Small token embedding dimension
    _PATCH: int = 16  # patch size


    def __init__(
            self,
            pretrained: bool = True,
            img_size: int = 224,
            tap_layers: tuple = (3, 7, 11),
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for DeiTSmallBackbone.  "
                "Install it with:  pip install timm"
            )

        assert len(tap_layers) == 3, "tap_layers must contain exactly 3 indices"
        self.tap_layers = tap_layers
        self.img_size = img_size
        self._h_patches = img_size // self._PATCH  # e.g. 14 for 224-px input
        self._w_patches = img_size // self._PATCH

        # ── Load DeiT-Small from timm ────────────────────────────────────────
        weights_key = "deit_small_patch16_224.fb_in1k" if pretrained else "deit_small_patch16_224"
        self._vit = timm.create_model(
            weights_key,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # remove classification head
        )
        # Freeze patch embedding & positional embedding (keep transformer body trainable)
        for p in self._vit.patch_embed.parameters():
            p.requires_grad_(False)

        D = self._DEIT_DIM  # 384

        # ── 1×1 projection convolutions for each tapped level ────────────────
        # Deep  (stride-16): D → C3  – identity-size, often no-op when C3==D
        self.proj_c = nn.Sequential(
            nn.Conv2d(D, self.C3, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.C3),
            nn.ReLU(inplace=True),
        )
        # Mid   (stride-8 after 2× upsample): D → C2
        self.proj_b = nn.Sequential(
            nn.Conv2d(D, self.C2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.C2),
            nn.ReLU(inplace=True),
        )
        # Shallow (stride-4 after 4× upsample): D → C1
        self.proj_a = nn.Sequential(
            nn.Conv2d(D, self.C1, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.C1),
            nn.ReLU(inplace=True),
        )

        # Weight init for projection layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    # ── helpers ───────────────────────────────────────────────────────────────

    def _tokens_to_map(self, tokens: Tensor, h: int, w: int) -> Tensor:
        """Reshape flat patch-token sequence → spatial feature map.

        tokens : (B, N, D)  where N = h*w (class token already removed)
        returns: (B, D, h, w)
        """
        B, N, D = tokens.shape
        assert N == h * w, f"Token count {N} doesn't match grid {h}×{w}"
        return tokens.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)


    def _run_vit_with_taps(self, x: Tensor):
        """Forward through the ViT, collecting patch tokens at tap_layers.

        Returns a list of three (B, h_patches, w_patches, D) tensors in the
        order [shallow, mid, deep].
        """
        import torch.nn.functional as F

        vit = self._vit

        # 1. Patch embedding → (B, N+1, D)  (+1 for class token)
        x = vit.patch_embed(x)

        # Prepend class token and add positional embedding
        B = x.shape[0]
        cls_token = vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = vit.pos_drop(x + vit.pos_embed)

        taps = {}
        # 2. Run transformer blocks, collecting outputs at requested indices
        for idx, block in enumerate(vit.blocks):
            x = block(x)
            if idx in self.tap_layers:
                # Strip class token → patch tokens only
                taps[idx] = x[:, 1:, :]  # (B, N, D)

        return [taps[i] for i in self.tap_layers]  # [shallow, mid, deep]


    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> List[Tensor]:
        import torch.nn.functional as F

        B, C, H, W = x.shape
        h = H // self._PATCH  # number of patch rows
        w = W // self._PATCH  # number of patch cols

        shallow_tok, mid_tok, deep_tok = self._run_vit_with_taps(x)

        # ── Deep branch: stride-16  (no spatial upsampling) ─────────────────
        deep_map = self._tokens_to_map(deep_tok, h, w)  # (B, D, h, w)
        feature_c = self.proj_c(deep_map)  # (B, C3, h,   w)

        # ── Mid branch: upsample 2× → stride-8 ─────────────────────────────
        mid_map = self._tokens_to_map(mid_tok, h, w)  # (B, D, h, w)
        mid_up = F.interpolate(mid_map, scale_factor=2,
                               mode='bilinear', align_corners=False)
        feature_b = self.proj_b(mid_up)  # (B, C2, 2h,  2w)

        # ── Shallow branch: upsample 4× → stride-4 ──────────────────────────
        sha_map = self._tokens_to_map(shallow_tok, h, w)  # (B, D, h, w)
        sha_up = F.interpolate(sha_map, scale_factor=4,
                               mode='bilinear', align_corners=False)
        feature_a = self.proj_a(sha_up)  # (B, C1, 4h,  4w)

        return [feature_a, feature_b, feature_c]


# ── BN_layer_deit ─────────────────────────────────────────────────────────────

class BN_layer_deit(nn.Module):
    """
    BN_layer adapted for DeiTSmallBackbone.

    Mirrors BN_layer_mobilenet exactly in structure:
      - conv1 path : x[0] (C1=96,  stride-4)  → OUT_C (384) at stride-16
                     via two consecutive stride-2 conv3×3
      - conv3 path : x[1] (C2=192, stride-8)  → OUT_C (384) at stride-16
                     via one stride-2 conv3×3
      - x[2]       : (C3=384, stride-16) passed through directly (no projection)
      - cat [l1, l2, x[2]] → (B, OUT_C*3=1152, H/16, W/16)
      - bn_layer (AttnBasicBlock stack, stride-2) → (B, BN_OUT_C=384, H/32, W/32)

    The channel arithmetic mirrors BN_layer_mobilenet 1-to-1 with the updated
    constants C1/C2/C3/OUT_C/BN_OUT_C that fit the DeiT-Small embedding size.
    """

    OUT_C: int = 384  # = C3 = DeiT hidden dim
    BN_OUT_C: int = 384  # output consumed by MobileNetV2Decoder-analogue

    def __init__(
            self,
            layers: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs,  # absorbs groups / width_per_group for API parity
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = 1
        self.base_width = 64
        self.dilation = 1

        C1, C2, C3 = (DeiTSmallBackbone.C1,
                      DeiTSmallBackbone.C2,
                      DeiTSmallBackbone.C3)
        OUT_C = self.OUT_C  # 384
        BN_OUT_C = self.BN_OUT_C  # 384

        self.relu = nn.ReLU(inplace=True)

        # ── conv1 path: C1 (96) at stride-4 → OUT_C (384) at stride-16 ─────
        # Two strided 3×3 convolutions (each ×2 downsample)
        self.conv1 = conv3x3(C1, OUT_C // 2, stride=2)  # 96  → 192, stride 4→8
        self.bn1 = norm_layer(OUT_C // 2)
        self.conv2 = conv3x3(OUT_C // 2, OUT_C, stride=2)  # 192 → 384, stride 8→16
        self.bn2 = norm_layer(OUT_C)

        # ── conv3 path: C2 (192) at stride-8 → OUT_C (384) at stride-16 ────
        self.conv3 = conv3x3(C2, OUT_C, stride=2)  # 192 → 384, stride 8→16
        self.bn3 = norm_layer(OUT_C)

        # ── x[2] path: C3=384=OUT_C at stride-16, no projection needed ──────

        # ── bn_layer: AttnBasicBlock stack on cat([l1, l2, x[2]]) ────────────
        # inplanes = OUT_C * 3 = 1152
        self.inplanes = OUT_C * 3  # 1152
        self.bn_layer = self._make_layer(BN_OUT_C, layers, stride=2)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * AttnBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * AttnBasicBlock.expansion, stride),
                norm_layer(planes * AttnBasicBlock.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(
            AttnBasicBlock(self.inplanes, planes, stride, downsample,
                           self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * AttnBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                AttnBasicBlock(self.inplanes, planes,
                               groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: List[Tensor]) -> Tensor:
        # x[0]: (B, 96,  H/4,  W/4)
        l1 = self.relu(self.bn1(self.conv1(x[0])))  # → (B, 192, H/8,  W/8)
        l1 = self.relu(self.bn2(self.conv2(l1)))  # → (B, 384, H/16, W/16)

        # x[1]: (B, 192, H/8,  W/8)
        l2 = self.relu(self.bn3(self.conv3(x[1])))  # → (B, 384, H/16, W/16)

        # x[2]: (B, 384, H/16, W/16) — no projection needed
        feature = torch.cat([l1, l2, x[2]], dim=1)  # → (B, 1152, H/16, W/16)

        output = self.bn_layer(feature)  # → (B,  384, H/32, W/32)
        return output.contiguous()

    def forward(self, x: List[Tensor]) -> Tensor:
        return self._forward_impl(x)


# ── Factory function ──────────────────────────────────────────────────────────

def deit_small_rd4ad(
        pretrained: bool = True,
        progress: bool = True,
        img_size: int = 224,
        tap_layers: tuple = (3, 7, 11),
        **kwargs: Any,
):
    backbone = DeiTSmallBackbone(
        pretrained=pretrained,
        img_size=img_size,
        tap_layers=tap_layers,
    )
    bn_layer = BN_layer_deit(layers=2, **kwargs)
    return backbone, bn_layer
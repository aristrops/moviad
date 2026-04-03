import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


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

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
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
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

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
        upsample: Optional[nn.Module] = None,
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
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
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

        if self.upsample is not None:
            identity = self.upsample(x)

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
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
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
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        feature_a = self.layer1(x)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
        #feature_d = self.layer4(feature_c)  # 64*64*64->128*32*32

        #x = self.avgpool(feature_d)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return [feature_c, feature_b, feature_a]

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
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

def de_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def de_resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def de_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

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

def de_wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
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
                   pretrained, progress, **kwargs)


def de_wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
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
                   pretrained, progress, **kwargs)


class MobileNetV2Decoder(ResNet):
    """
    Decoder paired with MobileNetV2Backbone + BN_layer_mobilenet.

    Subclasses ResNet to reuse _make_layer (with deconv2x2 upsampling shortcuts)
    and BasicBlock (with transposed-conv stride-2 path) unchanged.  The only
    difference from de_resnet18/34 is the channel sequence, which mirrors the
    MobileNetV2 encoder in reverse:

        BN_layer_mobilenet output  →  (B, 192, H/32, W/32)
        layer1: 192 → 96ch, ×2 up →  (B,  96, H/16, W/16)  ← mirrors C3
        layer2:  96 → 32ch, ×2 up →  (B,  32, H/8,  W/8)   ← mirrors C2
        layer3:  32 → 24ch, ×2 up →  (B,  24, H/4,  W/4)   ← mirrors C1

    Returns [feat_c (24ch), feat_b (32ch), feat_a (96ch)], matching the
    finest-first convention of de_resnet (_forward_impl returns
    [feature_c, feature_b, feature_a]).

    Channel constants are imported from the encoder file via the same
    integer literals used in BN_layer_mobilenet, keeping both sides in sync.
    """

    # Mirror the encoder's channel constants (MobileNetV2Backbone.C1/C2/C3)
    _C1: int = 24  # stride-4  features
    _C2: int = 32  # stride-8  features
    _C3: int = 96  # stride-16 features
    _BN_OUT_C: int = 192  # BN_layer_mobilenet.BN_OUT_C

    def __init__(
            self,
            layers: int = 2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs,  # absorbs unused kwargs (e.g. groups, width_per_group) for API parity
    ) -> None:
        # Bypass ResNet.__init__ entirely: we set up everything ourselves using
        # the inherited _make_layer, which only depends on self.inplanes,
        # self.dilation, self.groups, self.base_width, and self._norm_layer.
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # _make_layer reads and mutates self.inplanes — start at BN_layer output
        self.inplanes = self._BN_OUT_C  # 192
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Three upsampling stages, each ×2, mirroring the three encoder projections
        self.layer1 = self._make_layer(BasicBlock, self._C3, layers, stride=2)  # 192→96
        self.layer2 = self._make_layer(BasicBlock, self._C2, layers, stride=2)  # 96→32
        self.layer3 = self._make_layer(BasicBlock, self._C1, layers, stride=2)  # 32→24

        # Weight init identical to ResNet / BN_layer_mobilenet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # _make_layer, _forward_impl, and forward are all inherited from ResNet unchanged.


def de_mobilenet_v2(layers: int = 2, **kwargs: Any) -> MobileNetV2Decoder:
    """
    Decoder counterpart to mobilenet_v2_rd4ad() in resnet.py.

    Usage::

        from resnet import mobilenet_v2_rd4ad
        from resnet_decoder import de_mobilenet_v2

        encoder, bn_layer = mobilenet_v2_rd4ad(pretrained=True)
        decoder = de_mobilenet_v2()

    The ``layers`` argument controls how many BasicBlocks are stacked per
    upsampling stage and should match the ``layers`` passed to
    BN_layer_mobilenet (default 2).
    """
    return MobileNetV2Decoder(layers=layers, **kwargs)


class DeiTSmallDecoder(ResNet):
    """
    Decoder paired with DeiTSmallBackbone + BN_layer_deit.

    Mirrors MobileNetV2Decoder exactly in structure, with updated channel
    constants that match the DeiT-Small encoder:

        BN_layer_deit output  →  (B, 384, H/32, W/32)   (BN_OUT_C = 384)
        layer1: 384 → 384ch, ×2 up  →  (B, 384, H/16, W/16)  ← mirrors C3
        layer2: 384 → 192ch, ×2 up  →  (B, 192, H/8,  W/8)   ← mirrors C2
        layer3: 192 →  96ch, ×2 up  →  (B,  96, H/4,  W/4)   ← mirrors C1

    Returns [feat_c (96ch), feat_b (192ch), feat_a (384ch)], matching the
    finest-first convention of de_resnet:
        _forward_impl returns [feature_c, feature_b, feature_a]

    Each upsampling stage uses the inherited BasicBlock with deconv2x2 when
    stride=2, so the decoder is a pure transposed-convolution mirror of the
    encoder pyramid — exactly as in de_resnet18 / MobileNetV2Decoder.
    """

    # Mirror encoder channel constants (from DeiTSmallBackbone / BN_layer_deit)
    _C1: int = 96  # stride-4  features
    _C2: int = 192  # stride-8  features
    _C3: int = 384  # stride-16 features  (= DeiT hidden dim)
    _BN_OUT_C: int = 384  # BN_layer_deit.BN_OUT_C

    def __init__(
            self,
            layers: int = 2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs,  # absorbs unused kwargs for API parity
    ) -> None:
        # Bypass ResNet.__init__ — set up everything via inherited _make_layer
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # _make_layer reads and mutates self.inplanes — start at BN_layer output
        self.inplanes = self._BN_OUT_C  # 384
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Three upsampling stages, each ×2 in spatial resolution
        self.layer1 = self._make_layer(BasicBlock, self._C3, layers, stride=2)  # 384→384
        self.layer2 = self._make_layer(BasicBlock, self._C2, layers, stride=2)  # 384→192
        self.layer3 = self._make_layer(BasicBlock, self._C1, layers, stride=2)  # 192→96

        # Weight init identical to ResNet / BN_layer_deit
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # _make_layer, _forward_impl, and forward are all inherited from ResNet.
    #
    # _forward_impl applies the three layers in order and returns:
    #   [feature_c (C1=96, stride-4), feature_b (C2=192, stride-8),
    #    feature_a (C3=384, stride-16)]
    # which is exactly the finest-first list that RD4AD's anomaly scoring
    # expects from the de_resnet decoders.


def de_deit_small(layers: int = 2, **kwargs: Any) -> DeiTSmallDecoder:
    """
    Decoder counterpart to deit_small_rd4ad() in resnet.py.

    Usage::

        from resnet import deit_small_rd4ad
        from resnet_decoder import de_deit_small

        encoder, bn_layer = deit_small_rd4ad(pretrained=True)
        decoder = de_deit_small()

        # typical RD4AD forward pass
        features   = encoder(images)           # [a, b, c]
        bottleneck = bn_layer(features)        # fused tensor
        dec_feats  = decoder(bottleneck)       # reconstructed pyramid [c', b', a']

    The ``layers`` argument controls the number of BasicBlocks stacked per
    upsampling stage and should match the ``layers`` passed to
    BN_layer_deit (default 2).
    """
    return DeiTSmallDecoder(layers=layers, **kwargs)
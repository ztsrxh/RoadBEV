import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from models.submodule import *
from .mobilenetv3 import ConvNormActivation, SqueezeExcitation
from .mobilenetv3 import _make_divisible


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class UpSampleBN(nn.Module):
    def __init__(self, skip_channel, cur_channel, out_channel, interploate=False):
        super(UpSampleBN, self).__init__()

        self.interploate = interploate
        self.conv = nn.Sequential(BasicConv(skip_channel+cur_channel, out_channel, stride=1, kernel_size=3, padding=1),
                                  BasicConv(out_channel, out_channel, stride=1, kernel_size=3, padding=1))
        if not self.interploate:
            self.up = BasicConv(cur_channel, cur_channel, deconv=True, relu=False, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, concat_with):
        if self.interploate:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        else:
            up_x = self.up(x)
        f = torch.cat([up_x, concat_with], dim=1)
        return self.conv(f)

class EfficientNetCla(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.num_classes = num_classes
        layers: List[nn.Module] = []

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                self.num_classes,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        return x

class EfficientNetFeature(nn.Module):
    def __init__(
        self,
        stereo,
        inverted_residual_setting: List[MBConvConfig],
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stereo = stereo
        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU)
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        self.l0 = layers[0]
        self.l1 = layers[1]
        self.l2 = layers[2]
        self.l3 = layers[3]
        self.l4 = layers[4]
        self.feat_channel = 64
        if self.stereo:
            self.final_conv = nn.Sequential(convbn(288, 288, 5, 1, 2, 1),
                                            nn.ReLU(inplace=True),
                                            convbn(288, self.feat_channel, 3, 1, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.feat_channel, self.feat_channel, kernel_size=1, padding=0,
                                                      stride=1, bias=False))
        else:
            self.final_conv = nn.Sequential(convbn(256, 128, 5, 1, 2, 1),
                                            nn.ReLU(inplace=True),
                                            convbn(128, self.feat_channel, 3, 1, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.feat_channel, self.feat_channel, kernel_size=1, padding=0, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:

        x = self.l0(x)      # 1/2
        l1 = self.l1(x)     # 1/2
        l2 = self.l2(l1)    # 1/4
        l3 = self.l3(l2)    # 1/8
        l4 = self.l4(l3)    # 1/16

        if self.stereo:
            #############  1/2
            l2_interp = F.interpolate(l2, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            l3_interp = F.interpolate(l3, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            l4_interp = F.interpolate(l4, [l1.size()[2], l1.size()[3]], mode='bicubic', align_corners=True)
            feat = torch.cat((l1, l2_interp, l3_interp, l4_interp), dim=1)
        else:
            #############  1/4
            l3_interp = F.interpolate(l3, [l2.size()[2], l2.size()[3]], mode='bicubic', align_corners=True)
            l4_interp = F.interpolate(l4, [l2.size()[2], l2.size()[3]], mode='bicubic', align_corners=True)
            feat = torch.cat((l2, l3_interp, l4_interp), dim=1)

        x = self.final_conv(feat)

        return x

def efficientnet_cla(inplanes, num_classes, **kwargs: Any) -> EfficientNetCla:
    # # b0
    width_mult = 1.0
    depth_mult = 1.0

    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, inplanes, inplanes, 1),
        bneck_conf(6, 3, 1, inplanes, 128, 2),
        bneck_conf(6, 5, 1, 128, num_classes, 2),
        bneck_conf(6, 3, 1, num_classes, num_classes, 2),
    ]

    model = EfficientNetCla(inverted_residual_setting, num_classes=num_classes, **kwargs)

    return model

def efficientnet_feature(stereo, **kwargs: Any) -> EfficientNetFeature:
    # b6
    width_mult = 1.8
    depth_mult = 2.6

    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        # bneck_conf(6, 5, 1, 80, 112, 3),
        # bneck_conf(6, 5, 2, 112, 192, 4),
        # bneck_conf(6, 3, 1, 192, 320, 1),
    ]

    model = EfficientNetFeature(stereo, inverted_residual_setting, **kwargs)

    return model


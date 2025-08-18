import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..utils import resize


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class AggregateLocalAttention(nn.Module):
    def __init__(self, input, out, bn_norm_layer=nn.BatchNorm2d):
        super(AggregateLocalAttention, self).__init__()
        self.conv0 = nn.Conv2d(input, out, 3, 1, 1, bias=True, groups=out)
        self.bn_norm = bn_norm_layer(out)
        self.conv1 = nn.Conv2d(out, out, 1, bias=True, groups=out)
        self.conv3 = nn.Conv2d(out, out, 3, 1, 1, bias=True, groups=out)
        self.conv5 = nn.Conv2d(out, out, 5, 1, 2, bias=True, groups=out)
        self.local2 = ConvBN(out, out, kernel_size=1)

    def forward(self, res):
        res = self.conv0(res)
        res_ori = res.clone()
        res = self.bn_norm(res)

        attn_0 = self.conv1(res)
        attn_1 = self.conv3(res)
        attn_2 = self.conv5(res)

        x = attn_0 + attn_1 + attn_2 + res
        x = self.local2(x)
        x = x * res_ori
        return x


@HEADS.register_module()
class lmarshead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ala1 = AggregateLocalAttention(self.in_channels[0] + self.in_channels[1], self.in_channels[0])
        self.ala2 = AggregateLocalAttention(self.in_channels[1] + self.in_channels[2], self.in_channels[1])
        self.ala3 = AggregateLocalAttention(self.in_channels[2] + self.in_channels[3], self.in_channels[2])
        self.final = nn.Conv2d(self.in_channels[0], self.num_classes, 1)

    def forward(self, inputs):
        t1, t2, t3, t4 = inputs
        d3 = resize(t4, size=t3.size()[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([t3, d3], dim=1)
        d3 = self.ala3(d3)

        d2 = resize(d3, size=t2.size()[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([t2, d2], dim=1)
        d2 = self.ala2(d2)

        d1 = resize(d2, size=t1.size()[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([t1, d1], dim=1)
        d1 = self.ala1(d1)

        out = self.final(d1)
        return out

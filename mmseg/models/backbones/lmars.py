import torch
from torch import nn
import torch.nn.functional as F
from ..builder import BACKBONES
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
import math
import warnings
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmengine.utils import to_2tuple
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn import Conv2d, build_activation_layer


class Upgraded_FFN(BaseModule):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(Upgraded_FFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        self.fc1 = nn.Linear(in_channels, feedforward_channels)
        self.pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = nn.Linear(feedforward_channels, in_channels)
        drop = nn.Dropout(ffn_drop)
        layers = [self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = self.fc1(x)
        out = nlc_to_nchw(out, hw_shape)
        out = self.pe_conv(out) + out
        out = nchw_to_nlc(out)
        out = self.layers(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class WindowMSA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, window_size[0])), nn.AdaptiveAvgPool2d((window_size[0], 1))
        self.norm = nn.LayerNorm(embed_dims)

        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.qk = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape
        hw_shape = int(pow(N, 0.5)), int(pow(N, 0.5))
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = nlc_to_nchw(x, hw_shape)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_ = x * x_w * x_h
        x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)

        qk = self.qk(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        g = (attn @ v).transpose(1, 2).reshape(B, N, C)
        g = self.proj(g)
        x = self.proj_drop(g)
        return x


class SqueezeWindowMSA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        shifted_query = query
        query_windows = self.window_partition(shifted_query)

        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        attn_windows = self.w_msa(query_windows)

        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        x = self.window_reverse(attn_windows, H_pad, W_pad)

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):

        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class TransformerEncoderLayer(BaseModule):
    def __init__(self, dim, window_size, num_heads, qkv_bias, qk_scale,
                 attn_drop_rate, drop_path_rate, drop_rate, act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

        self.attn = SqueezeWindowMSA(
            embed_dims=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        self.ffn = Upgraded_FFN(
            embed_dims=dim,
            feedforward_channels=int(dim * 4),
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = x + self.attn(self.norm(x), hw_shape)
        x = x + self.ffn(self.norm(x), hw_shape, identity=x)

        return x


@BACKBONES.register_module()
class lmars(BaseModule):
    def __init__(self, input_channels=3,
                 c_list=[8, 16, 32, 64],
                 num_layers=[3, 4, 6, 3],
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 init_cfg=None,
                 window_size=7,
                 num_heads=(1, 2, 4, 8), qkv_bias=True, qk_scale=None,
                 attn_drop_rate=0., drop_path_rate=0.1, drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 ):
        super().__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        cur = 0
        self.layers = ModuleList()
        self.num_layers = num_layers
        self.out_indices = out_indices

        for i, num_layer in enumerate(num_layers):
            embed_dims_i = c_list[i]
            patch_embed = PatchEmbed(
                in_channels=input_channels,
                embed_dims=embed_dims_i,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(dim=embed_dims_i,
                                        window_size=window_size,
                                        num_heads=num_heads[i],
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=dpr[cur + idx],
                                        drop_rate=drop_rate) for idx in range(num_layer)
            ])
            input_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            self.layers.append(ModuleList([patch_embed, layer]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)

            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

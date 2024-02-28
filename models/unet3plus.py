import torch
import torch.nn as nn
import numpy as np
from typing import List


def u3pblock(in_ch, out_ch, num_block=2, k=3, pad=1, down_sample=False):
    m = []
    if down_sample:
        m.append(nn.MaxPool2d(kernel_size=2))
    for _ in range(num_block):
        m += [
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        in_ch = out_ch
    return nn.Sequential(*m)


# 编码器到解码器的层
def enc2dec_layer(in_ch, out_ch, scale):
    m = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    m.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*m)


# 解码器到解码器的连接
def dec2dec_layer(in_ch, out_ch, scale, fast_up=True):
    up = [nn.Upsample(scale_factor=scale, mode='bilinear',
                      align_corners=True) if scale != 1 else nn.Identity]
    m = [u3pblock(in_ch, out_ch, num_block=1)]

    if fast_up:
        m += up
    else:
        m = up + m
    return nn.Sequential(*m)


class FullScaleSkipConnect(nn.Module):
    def __init__(self,
                 en_channels,
                 en_scales,
                 num_dec,
                 skip_ch=64,
                 dec_scales=None,
                 bottom_dec_ch=1024,
                 dropout=0.3,
                 fast_up=True
                 ):
        super(FullScaleSkipConnect, self).__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)
        # encoder maps to decoder maps connections
        self.enc2dec_layers = nn.ModuleList()
        for en_ch, scale in zip(en_channels, en_scales):
            self.enc2dec_layers.append(enc2dec_layer(en_ch, skip_ch, scale))

        # decoder maps to decoder maps connections
        self.dec2dec_layers = nn.ModuleList()
        if dec_scales is None:
            dec_scales = []
            for ii in reversed(range(num_dec)):
                dec_scales.append(2 ** (ii + 1))
        for ii, scale in enumerate(dec_scales):
            dec_ch = bottom_dec_ch if ii == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_ch, skip_ch, scale, fast_up=fast_up))

        self.dropout = nn.Dropout(dropout)
        self.fuse_layer = u3pblock(concat_ch, concat_ch)

    def forward(self, en_maps, dec_maps=None):
        out = []
        for en_map, layer in zip(en_maps, self.enc2dec_layers):
            out.append(layer(en_map))
        if dec_maps is not None and len(dec_maps) > 0:
            for dec_map, layer in zip(dec_maps, self.dec2dec_layers):
                out.append(layer(dec_map))

        return self.fuse_layer(self.dropout(torch.cat(out, dim=1)))  # 320


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024), num_block=2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for ii, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            self.layers.append(u3pblock(in_ch, out_ch, num_block, down_sample=ii > 0))
        self.channels = channels

    def forward(self, x):
        encoder_out = []
        for layer in self.layers:
            x = layer(x)
            encoder_out.append(x)  # 各层输出特征
        return encoder_out


class Decoder(nn.Module):  # 解码器由全连接层组成
    def __init__(self,
                 en_channels=(64, 128, 256, 512, 1024),
                 skip_ch=64, dropout=0.3, fast_up=True):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleDict()
        en_channels = en_channels[::-1]  # 1024 512 256 128 64
        num_en_ch = len(en_channels)
        for ii in range(num_en_ch):
            if ii == 0:
                # 第一个解码层输出是最后一个编码层输入的恒等映射
                self.decoders['decoder1'] = nn.Identity()
                continue
            self.decoders[f'decoder{ii+1}'] = FullScaleSkipConnect(
                en_channels[ii:],
                en_scales=2 ** np.arange(0, num_en_ch-ii),  # 1, 2, 4, 8
                num_dec=ii,
                skip_ch=skip_ch,
                bottom_dec_ch=en_channels[0],
                dropout=dropout,
                fast_up=fast_up
            )

    def forward(self, enc_map_list: List[torch.Tensor]):
        dec_map_list = []
        enc_map_list = enc_map_list[::-1]
        for ii, layer_ley in enumerate(self.decoders):
            layer = self.decoders[layer_ley]
            if ii == 0:
                dec_map_list.append(layer(enc_map_list[0]))
                continue
            dec_map_list.append(layer(enc_map_list[ii:], dec_map_list))

        return dec_map_list


class Unet3plus(nn.Module):
    def __init__(self,
                 num_classes=1,
                 skip_ch=64,
                 aux_losses=2,
                 encoder: Encoder = None,
                 channels=(3, 64, 128, 256, 512, 1024),
                 dropout=0.3,
                 transpose_final=False,
                 use_cgm=False,
                 fast_up=True):
        super(Unet3plus, self).__init__()
        self.encoder = Encoder(channels) if encoder is None else encoder
        channels = self.encoder.channels  # (3, 64, 128, 256, 512, 1024)
        num_decoders = len(channels) - 1
        decoder_ch = skip_ch * num_decoders

        self.decoder = Decoder(self.encoder.channels[1:],
                               skip_ch=skip_ch,
                               dropout=dropout,
                               fast_up=fast_up)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(channels[-1], 2, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        ) if use_cgm and num_classes <= 2 else None

        if transpose_final:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(decoder_ch, num_classes, kernel_size=4, stride=2, padding=1,
                                   bias=False),
            )
        else:
            self.head = nn.Conv2d(decoder_ch, num_classes, 3, padding=1)


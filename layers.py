import torch.nn as nn
from torch.nn.init import kaiming_normal_, xavier_normal_


def weights_init(conv_weights, init, activate):
    if init is None:
        return
    if init == 'kaiming':
        if hasattr(activate, 'negative_slope'):
            kaiming_normal_(conv_weights, a=activate.negative_slope)
        else:
            kaiming_normal_(conv_weights, a=0)
    elif init == 'xavier_normal':
        xavier_normal_(conv_weights)
    return


def conv(in_channels, out_channels, ks, s, pad=0, init='kaiming',
         activate=nn.ReLU(), batch_norm=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=s, padding=pad)
    ]
    weights_init(layers[0].weight, init, activate)
    if activate is not None:
        layers.append(activate)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, ks, s, pad=0, out_pad=0, init='kaiming',
           activate=nn.ReLU(), batch_norm=False):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, ks, s, pad, out_pad)
    ]
    weights_init(layers[0].weight, init, activate)
    if activate is not None:
        layers.append(activate)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 ks=3, s=1, p=None,
                 weight_init='kaiming',
                 activate=nn.ReLU(),
                 batch_norm=False,
                 ):
        super(ResidualBlock, self).__init__()
        self.activate = activate
        if out_channels is None:
            out_channels = in_channels // s
        if s == 1:
            self.shortcut = nn.Identity()
        layers = [
            conv(in_channels, out_channels, ks, 1,
                 p if p is not None else (ks-1)//2,
                 weight_init, activate, batch_norm),
            conv(in_channels, out_channels, ks, s,
                 p if p is not None else (ks-1)//2,
                 None, None, batch_norm)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        short_cut = self.shortcut(x)
        out = self.layers(x)

        return self.activate(short_cut + out)

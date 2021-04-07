import torch.nn as nn

from .base import GANBase


def _deconv(
    in_c, out_c, kernel, stride, pad, output_pad, bn=True, act="lrelu"
):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_pad)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c, momentum=0.8))
    if act == "lrelu":
        layers.append(nn.LeakyReLU(0.2))
    elif act == "relu":
        layers.append(nn.ReLU())
    elif act is not None:
        raise NotImplementedError
    return nn.Sequential(*layers)


def _deconv2(
    in_c, out_c, kernel, stride, pad, upscale=2, upsize=None,
    bn=True, act="lrelu"
):
    if upsize is not None:
        layers = [nn.Upsample(upsize)]
    else:
        layers = [nn.Upsample(scale_factor=upscale)]
    layers.append(nn.Conv2d(in_c, out_c, kernel, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(out_c, momentum=0.8))
    if act == "lrelu":
        layers.append(nn.LeakyReLU(0.2))
    elif act == "relu":
        layers.append(nn.ReLU())
    elif act is not None:
        raise NotImplementedError
    return nn.Sequential(*layers)


def _conv(
    in_c, out_c, kernel, stride, pad, bn=True, act="lrelu", dropout=None
):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, pad)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c, momentum=0.8))
    if act == "lrelu":
        layers.append(nn.LeakyReLU(0.2))
    elif act == "relu":
        layers.append(nn.ReLU())
    elif act is not None:
        raise NotImplementedError
    if dropout is not None:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class DCGAN(GANBase):

    def __init__(
        self, img_shape, latent_size, z_dist="uniform", z_std=1,
        gen_type=1, disc_type=1
    ):
        super().__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.z_dist = z_dist
        self.z_std = z_std

        if gen_type == 1:
            self.generator = nn.Sequential(
                nn.Linear(latent_size, 3*3*128),
                nn.Unflatten(1, (128, 3, 3)),
                nn.BatchNorm2d(128, momentum=0.8),
                _deconv2(128, 64, 3, 1, 1, upsize=(7, 7),
                         bn=True, act="lrelu"),
                _deconv2(64, 32, 3, 1, 1, 2, bn=True, act="lrelu"),
                _deconv2(32, 16, 3, 1, 1, 2, bn=False, act=None),
                _conv(16, 1, 3, 1, 1, False, None),
                nn.Tanh()
            )
        else:
            self.generator = nn.Sequential(
                nn.Linear(latent_size, 3*3*128),
                nn.Unflatten(1, (128, 3, 3)),
                nn.BatchNorm2d(128, momentum=0.8),
                _deconv(128, 64, 3, 2, 0, 0, True, "lrelu"),
                _deconv(64, 32, 3, 2, 1, 1, True, "lrelu"),
                _deconv(32, 16, 3, 2, 1, 1, True, "lrelu"),
                _conv(16, 1, 3, 1, 1, False, None),
                nn.Tanh()
            )

        # # 这个效果不错，先保存一下
        # 效果更好
        if disc_type == 1:
            self.discriminator = nn.Sequential(
                _conv(1, 16, 3, 2, 1, False, "lrelu", 0.25),    # --> 14x14
                _conv(16, 32, 3, 2, 1, False, "lrelu", 0.25),   # --> 7x7
                _conv(32, 64, 3, 2, 1, False, "lrelu", 0.25),   # --> 4x4
                _conv(64, 128, 3, 2, 1, False, "lrelu", 0.25),  # --> 2x2
                nn.Flatten(),
                nn.Linear(128 * 4, 1)
            )
        elif disc_type == 2:
            self.discriminator = nn.Sequential(
                _conv(1, 16, 3, 2, 1, False, "lrelu", 0.25),    # --> 14x14
                _conv(16, 32, 3, 2, 1, False, "lrelu", 0.25),   # --> 7x7
                _conv(32, 64, 3, 2, 1, False, "lrelu", 0.25),   # --> 4x4
                _conv(64, 128, 3, 2, 1, False, "lrelu", 0.25),  # --> 2x2
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 1)
            )
        else:
            self.discriminator = nn.Sequential(
                _conv(1, 16, 3, 2, 1, False, "lrelu", 0.25),    # --> 14x14
                _conv(16, 32, 3, 2, 1, False, "lrelu", 0.25),   # --> 7x7
                _conv(32, 64, 3, 2, 1, False, "lrelu", 0.25),   # --> 4x4
                _conv(64, 128, 3, 2, 1, False, "lrelu", 0.25),  # --> 2x2
                _conv(128, 1, 3, 1, 1, False, None),            # --> 2x2
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

    def discriminate(self, x):
        return self.discriminator(x)

    def generate(self, z):
        return self.generator(z)

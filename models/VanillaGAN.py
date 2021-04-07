from math import prod

import torch.nn as nn

from .base import GANBase


class VanillaGAN(GANBase):

    def __init__(self, img_shape, latent_size, z_dist, z_std):
        super().__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.z_dist = z_dist
        self.z_std = z_std

        in_c = prod(img_shape)

        self.flatten = nn.Flatten()
        self.discriminator = nn.Sequential(
            nn.Linear(in_c, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.generator = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.BatchNorm1d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, in_c),
            nn.Tanh()
        )

    def discriminate(self, x):
        return self.discriminator(self.flatten(x))

    def generate(self, z):
        x = self.generator(z)
        x = x.reshape(-1, *self.img_shape)
        # 这样做实际上损失了一定的效率，但可以保证模型代码的一致性
        return x

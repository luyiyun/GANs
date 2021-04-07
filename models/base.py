from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANBase(nn.Module):

    def discriminate(self, x):
        raise NotImplementedError

    def generate(self, z):
        raise NotImplementedError

    def z_sample(self, size):
        if self.z_dist == "normal":
            return torch.normal(
                torch.zeros(size),
                torch.full(size, self.z_std)
            )
        elif self.z_dist == "uniform":
            z = (torch.rand(size) - 0.5) / 2 * sqrt(3) * self.z_std
            return z
        else:
            raise NotImplementedError

    @staticmethod
    def disc_criterion(pred_gen, pred_tru):
        target_gen = torch.zeros_like(pred_gen)
        target_tru = torch.ones_like(pred_tru)
        target = torch.cat([target_gen, target_tru])
        pred = torch.cat([pred_gen, pred_tru])
        return F.binary_cross_entropy_with_logits(pred, target)

    @staticmethod
    def gene_criterion(pred_gen):
        target_gen = torch.ones_like(pred_gen)
        loss = F.binary_cross_entropy_with_logits(pred_gen, target_gen)
        return loss

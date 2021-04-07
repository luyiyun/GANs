import os
import sys
import json

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

try:
    import wandb
    WANDB_FLAG = True
except ModuleNotFoundError:
    WANDB_FLAG = False

import models as M


class LossesMetric:

    def __init__(self, keys=["disc", "gene"]):
        self._keys = keys
        self._totals = {k: 0. for k in keys}
        self._counts = {k: 0 for k in keys}

    def add(self, losses, bs):
        for k, v in losses.items():
            self._totals[k] = self._totals[k] + v.item() * bs
            self._counts[k] = self._counts[k] + bs

    def value(self):
        if hasattr(self, "_values"):
            return self._values
        self._values = {k: self._totals[k] / self._counts[k]
                        for k in self._keys}
        return self._values


def generate_gan(net, Z=None, num=100, nrow=10, device="cuda:0"):
    device = torch.device(device)
    net = net.to(device)

    if Z is None:
        Z = net.z_sample((num, net.latent_size)).to(device).float()
    with torch.no_grad():
        imgs_tensor = net.generate(Z)
    imgs_tensor = make_grid(imgs_tensor, nrow=nrow, normalize=True)
    return imgs_tensor


def train_gan(
    net, tr_loader, va_loader=None,
    epochs=100, device="cuda:0", disc_lr=0.001, gene_lr=0.001,
    disc_iter=1, gene_iter=1, writer=None
):
    device = torch.device(device)

    net = net.to(device)
    disc_optimizer = optim.Adam(
        net.discriminator.parameters(),
        lr=disc_lr, betas=(0.5, 0.999)
    )
    gene_optimizer = optim.Adam(
        net.generator.parameters(),
        lr=gene_lr, betas=(0.5, 0.999)
    )

    hist = {"epoch": [], "phase": []}
    for e in tqdm(range(epochs), "Epoch: "):
        losses_cache = LossesMetric()
        for X, _ in tqdm(tr_loader, "Batch Train: ", leave=False):
            X = X.to(device).float()

            # discriminator training
            net.discriminator.train()
            net.generator.eval()
            for _ in range(disc_iter):
                Z = net.z_sample(
                    (X.size(0), net.latent_size)).to(device).float()
                disc_optimizer.zero_grad()
                with torch.no_grad():
                    X_gen = net.generate(Z)
                with torch.enable_grad():
                    pred_gen = net.discriminate(X_gen)
                    pred_tru = net.discriminate(X)
                    disc_loss = net.disc_criterion(pred_gen, pred_tru)
                disc_loss.backward()
                disc_optimizer.step()

            # generator training
            net.discriminator.eval()
            net.generator.train()
            for _ in range(gene_iter):
                Z = net.z_sample(
                    (X.size(0), net.latent_size)).to(device).float()
                gene_optimizer.zero_grad()
                with torch.enable_grad():
                    X_gen = net.generate(Z)
                    pred_gen = net.discriminate(X_gen)
                    gene_loss = net.gene_criterion(pred_gen)
                gene_loss.backward()
                gene_optimizer.step()

            losses_cache.add({"disc": disc_loss, "gene": gene_loss}, X.size(0))

        losses = losses_cache.value()
        hist["epoch"].append(e)
        hist["phase"].append("train")
        for k, v in losses.items():
            hist.setdefault(k, []).append(v)

        if writer is not None:
            losses["epoch"] = e
            writer.log(losses)
            # 生成一定数量的示例图像，并进行展示
            if (e + 1) % 5 == 0:
                Z = net.z_sample((16, net.latent_size)).to(device).float()
                imgs_tensor = net.generate(Z)
                imgs_tensor = make_grid(imgs_tensor, nrow=8, normalize=True)
                img = to_pil_image(imgs_tensor)
                wandb_imgs = wandb.Image(img)
                writer.log({"generated": wandb_imgs, "epoch": e})

    return net, hist


MODELS = {
    "VanillaGAN": M.VanillaGAN,
    "DCGAN": M.DCGAN,
}


def main():
    dat_root = "/home/stat-luyiyun/Datasets/"
    if len(sys.argv) == 1:
        conf_fn = "./configs/vanilla_gan.yaml"
    else:
        arg = sys.argv[1]
        if arg == "ls":
            for fn in os.listdir("./configs"):
                if fn.endswith(".yaml"):
                    print(fn[:-5])
            return
        else:
            conf_fn = os.path.join("./configs/", sys.argv[1]+".yaml")
    with open(conf_fn, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)

    writer = None
    if WANDB_FLAG:
        writer = wandb.init(
            dir="/home/stat-luyiyun/wandb", config=conf,
            project="GANs", name=conf["name"]
        )
        conf = writer.config

    transfer = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    tr_dat = MNIST(dat_root, train=True, transform=transfer)
    tr_loader = DataLoader(
        tr_dat, batch_size=conf["bs"], shuffle=True,
        num_workers=conf["nw"], pin_memory=True
    )

    # model
    net = MODELS[conf["name"]](img_shape=(1, 28, 28), **conf["params"])

    # training
    net, hist = train_gan(
        net, tr_loader, None, conf["epochs"],
        conf["device"], conf["disc_lr"], conf["gene_lr"],
        writer=writer
    )

    # generation
    imgs_tensor = generate_gan(net, device=conf["device"])
    save_image(imgs_tensor, "./imgs/%s.png" % conf["name"])

    if os.path.exists("./results"):
        save_dir = os.path.join("./results", conf["name"])
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "hist.json"), "w") as f:
            json.dump(hist, f)
        torch.save(net.state_dict(), os.path.join(save_dir, "model.pth"))


if __name__ == "__main__":
    main()

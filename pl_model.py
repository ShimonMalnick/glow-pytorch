import os
from easydict import EasyDict
import pytorch_lightning as pl
import torch.nn as nn
from model import Block
import torch
from train import calc_loss, calc_z_shapes
from pytorch_lightning.loggers import WandbLogger
from utils import get_args, make_exp_dir, save_dict_as_json, get_dataloader
from torchvision.utils import make_grid


class LitGlow(pl.LightningModule):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True, lr=1e-4, n_bits=5, img_size=128,
            n_sample=20, temp=0.7, **kwargs
    ):
        super().__init__()
        self.lr = lr
        self.n_bits = n_bits
        self.n_bins = 2.0 ** n_bits
        self.img_size = img_size
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))
        self.automatic_optimization = False
        z_sample = []
        z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
        for z in z_shapes:
            z_new = torch.randn(n_sample, *z) * temp
            z_sample.append(z_new)
        self.z_sample = z_sample
        self.save_hyperparameters()
        self.extra_args = kwargs

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        opt = self.optimizers()
        opt.zero_grad()
        image, _ = batch
        image = image * 255

        if self.n_bits < 8:
            image = torch.floor(image / 2 ** (8 - self.n_bits))

        image = image / self.n_bins - 0.5

        if batch_idx == 0:
            with torch.no_grad():
                log_p, logdet, _ = self.forward(
                    image + torch.rand_like(image) / self.n_bins
                )
                return {"init_loss": torch.tensor(0.0).type_as(image)}

        else:
            log_p, logdet, _ = self.forward(image + torch.rand_like(image) / self.n_bins)

        logdet = logdet.mean()


        loss, log_p, log_det = calc_loss(log_p, logdet, self.img_size, self.n_bins)
        self.manual_backward(loss)
        opt.step()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/log_p", log_p, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/logdet", log_det, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if (batch_idx + 1) % 100 == 0:
            with torch.no_grad():
                image = make_grid(self.reverse(self.z_sample), normalize=True, nrow=10, range=(-0.5, 0.5))
                self.logger.log_image(key=f'samples', images=[image])
                # save_image(
                #     self.reverse(self.z_sample).cpu().data,
                #     f'experiments/{self.extra_args["exp_name"]}/samples/{str(self.global_step + 1).zfill(6)}.png',
                #     normalize=True,
                #     nrow=10,
                #     range=(-0.5, 0.5))

        return {'train_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def run_training(args: EasyDict):
    #steup model and data loader
    model = LitGlow(in_channel=3, **args)
    train_dl = get_dataloader(args.path, args.batch, args.img_size, args.num_workers)

    # setup trainer and its arguments
    logger = WandbLogger(project="Glow", entity="malnick")
    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.exp_name, 'checkpoints'), save_top_k=-1,
                                                 every_n_train_steps=10000)
    trainer = pl.Trainer(logger=logger, default_root_dir=os.path.join(args.exp_name, 'checkpoints'), accelerator='ddp',
                         num_nodes=1, gpus=args.devices, callbacks=[ckpt_callback])
    trainer.fit(model, train_dataloaders=train_dl)


if __name__ == '__main__':
    args = get_args()
    args.exp_name = make_exp_dir(args.exp_name, exist_ok=True)
    print(args)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    run_training(args)

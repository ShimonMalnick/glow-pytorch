from utils import get_args, make_exp_dir, save_dict_as_json, get_dataloader
from model import Glow
import torch
from pytorch_lightning.lite import LightningLite
from easydict import EasyDict
from typing import Tuple
from torchvision.utils import save_image
from train import calc_z_shapes, calc_loss


def get_model_and_opt(args: EasyDict) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.opt_path:
        optimizer.load_state_dict(torch.load(args.opt_path, map_location=lambda storage, loc: storage))
    return model, optimizer


class Lite(LightningLite):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, args: EasyDict):
        model, optimizer = get_model_and_opt(args)
        model, optimizer = self.setup(model, optimizer)  # scale
        print("Loaded model and optimizer")
        dataloader = get_dataloader(args.path, args.batch, args.img_size, args.num_workers)
        dataloader = self.setup_dataloaders(dataloader)  # scale
        print("Built Dataloader")
        num_epcohs = max(1, args.iter // len(dataloader.dataset))
        model.train()

        n_bins = 2.0 ** args.n_bits

        z_sample = []
        z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
        for z in z_shapes:
            z_new = torch.randn(args.n_sample, *z) * args.temp
            z_sample.append(z_new)

        for e in range(num_epcohs):
            for i, batch in enumerate(dataloader):
                image, _ = batch
                image = image * 255

                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5

                if i == 0 and e == 0:
                    with torch.no_grad():
                        log_p, logdet, _ = model.module(
                            image + torch.rand_like(image) / n_bins
                        )
                        continue

                else:
                    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                logdet = logdet.mean()

                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                model.zero_grad()
                self.backward(loss)
                warmup_lr = args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                if (i + 1) % 100 == 0:
                   print(f"Epoch: {e + 1}; Iter: {i + 1}; Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")
                   with torch.no_grad():
                       save_image(
                            model.reverse(z_sample).cpu().data,
                            f'experiments/{args.exp_name}/samples/{str(i + 1).zfill(6)}.png',
                            normalize=True,
                            nrow=10,
                            range=(-0.5, 0.5))

                if (i + 1) % 10000 == 0:
                    torch.save(
                        model.state_dict(), f'experiments/{args.exp_name}/checkpoints/model_epc_{str(e + 1).zfill(3)}_itr_{str(i + 1).zfill(6)}.pt'
                    )
                    torch.save(
                        optimizer.state_dict(),
                        f'experiments/{args.exp_name}/checkpoints/optim_{str(i + 1).zfill(6)}.pt'
                    )


def run_main():
    args = get_args()
    args.exp_name = make_exp_dir(args.exp_name, exist_ok=True)
    print(args)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    Lite(accelerator='gpu', strategy='ddp', devices=args.devices, num_nodes=1).run(args)


if __name__ == '__main__':
    run_main()


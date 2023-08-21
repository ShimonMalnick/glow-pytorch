from tqdm import tqdm
from math import log
from utils import get_args, save_dict_as_json, save_model_optimizer
import torch
from torch import nn, optim
from torchvision import utils
from model import Glow
from utils import sample_data
from typing import List, Tuple
import wandb
import os
from time import time


def make_train_exp_dir(exp_name, exist_ok=False, dir_name="train") -> str:
    base_path = os.path.join("experiments", dir_name, exp_name)
    os.makedirs(f'{base_path}/checkpoints', exist_ok=exist_ok)
    os.makedirs(f'{base_path}/samples', exist_ok=exist_ok)
    return os.path.join(dir_name, exp_name)


def calc_z_shapes(n_channel, input_size, n_flow, n_block) -> List[Tuple]:
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


# comment on the repo from https://github.com/rosinality/glow-pytorch/issues/13
def calc_loss_changed(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet.mean() + log_p.mean()
    return (
        -loss / (log(2) * n_pixel),
        log_p.mean() / (log(2) * n_pixel),
        logdet.mean() / (log(2) * n_pixel)
    )


def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch, args.img_size, data_split=args.data_split,
                               training_labels=args.training_labels))
    print("Loaded Dataset", flush=True)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    cur = time()
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
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
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()
            if not args.debug_timing:
                wandb.log({"loss": loss.item(),
                           "log_p": log_p.item(),
                           "log_det": log_det.item(),
                           "prob": log_p.item() + log_det.item()})
                pbar.set_description(
                    f"iter: {i + 1};Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
                )

                if i % 100 == 0:
                    print(f'Avg time after {i + 1} iterations: {(time() - cur) / (i + 1):.5f} seconds')
                    cur_image_name = f'experiments/{args.exp_name}/samples/{str(i + 1).zfill(6)}.png'
                    with torch.no_grad():
                        utils.save_image(
                            model_single.reverse(z_sample).cpu().data,
                            cur_image_name,
                            normalize=True,
                            nrow=10,
                            range=(-0.5, 0.5),
                        )
                    wandb.log({"samples": wandb.Image(cur_image_name)})

                if i % 10000 == 0:
                    save_model_optimizer(args, i, model, optimizer)
            else:
                if i % 100 == 0:
                    run_time = time() - cur
                    print(f'Avg time after {i + 1} iterations: {run_time / (i + 1):.5f} seconds')


def evaluate(args, eval_model):
    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    print("*" * 30)
    print(z_shapes)
    print("*" * 30)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    file_name = f'experiments/{args.exp_name}/{args.sample_name}'
    if not args.sample_name:
        file_name += 'eval'
    file_name += ".png"
    with torch.no_grad():
        utils.save_image(
            eval_model.reverse(z_sample).cpu().data,
            file_name,
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    args.exp_name = make_train_exp_dir(args.exp_name)
    wandb.init(project="Glow-Train", entity="malnick", name=args.exp_name, config=args)
    debug_timing_mode = False
    args.debug_timing = debug_timing_mode
    print(args)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )

    model = nn.DataParallel(model_single)
    # model = model_single
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage))
    model = model.to(device)
    print("Loaded Model successfully", flush=True)
    if 'eval' in args and args.eval:
        evaluate(args, model_single)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if args.opt_path:
            optimizer.load_state_dict(torch.load(args.opt_path, map_location=lambda storage, loc: storage))
            print("Loaded Optimizer successfully", flush=True)
        train(args, model, optimizer)

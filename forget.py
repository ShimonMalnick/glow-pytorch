import json
import os
from glob import glob
from time import time

from easydict import EasyDict
from torchvision import transforms
from model import Glow
import torch
from utils import get_args, load_model, quantize_image, sample_data, make_exp_dir, save_dict_as_json, \
    save_model_optimizer, compute_bpd, gather_jsons
from torch.utils.data import Subset, Dataset, DataLoader
from typing import Iterator, List, Callable
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from train import calc_loss
import wandb
import re
import torchvision.datasets as vision_datasets


def get_forget_dataset(args, additional_tranform=None, images_path=None, max_size=False) -> Dataset:
    all_transforms = [Resize((args.img_size, args.img_size)), RandomHorizontalFlip(), ToTensor(),
                      lambda img: quantize_image(img, args.n_bits)]
    if additional_tranform is not None:
        all_transforms.append(additional_tranform)
    transform = Compose(all_transforms)
    if images_path is None:
        images_path = args.forget_path
    initial_ds = ImageFolder(images_path, transform=transform)
    if max_size:
        args.forget_size = len(initial_ds)
    assert args.forget_size <= len(initial_ds)
    ds = Subset(initial_ds, [i % args.forget_size for i in range(max(args.batch, args.forget_size))])
    return ds


def get_forget_dataloader(args, additional_tranform=None, images_path=None) -> Iterator:
    ds = get_forget_dataset(args, additional_tranform, images_path)
    return sample_data(args.forget_path, args.batch, args.img_size, dataset=ds)


def forget(args, regular_iter: Iterator, forget_iter: Iterator, model: Glow, device, forget_optimizer,
           regular_optimizer):
    n_bins = 2 ** args.n_bits
    for i in range(args.iter):
        if (i + 1) % args.forget_every == 0:
            print("Forget batch")
            loss_sign = -1.0
            loss_name = "forget"
            cur_batch, _ = next(forget_iter)
            optimizer = forget_optimizer
        else:
            print("regular batch")
            loss_sign = 1.0
            loss_name = "regular"
            cur_batch, _ = next(regular_iter)
            optimizer = regular_optimizer
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss_sign * loss
        if torch.abs(loss).item() > 1e6:
            print("Loss is too large, breaking after {} iterations".format(i), flush=True)
            break
        wandb.log({f"{loss_name}_loss": loss.item(),
                   f"{loss_name}_log_p": log_p.item(),
                   f"{loss_name}_log_det": log_det.item(),
                   f"{loss_name}_prob": log_p.item() + log_det.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; "
        )

        if (i + 1) % args.save_every == 0:
            torch.save(
                model.state_dict(), f'experiments/{args.exp_name}/checkpoints/model_{str(i + 1).zfill(6)}.pt'
            )
            torch.save(
                forget_optimizer.state_dict(),
                f'experiments/{args.exp_name}/checkpoints/forget_optim_{str(i + 1).zfill(6)}.pt'
            )
            torch.save(
                regular_optimizer.state_dict(),
                f'experiments/{args.exp_name}/checkpoints/regular_optim_{str(i + 1).zfill(6)}.pt'
            )


def forget_baseline(args, model, device, optimizer):
    if args.forget_noise is not None and args.forget_noise > 0:
        additional_transform = lambda img: torch.clamp(img + torch.randn_like(img) * args.forget_noise, -0.5, 0.5)
    else:
        additional_transform = None
    data_iter: Iterator = get_forget_dataloader(args, additional_tranform=additional_transform)
    n_bins = 2 ** args.n_bits
    for i in range(args.iter):
        cur_batch, _ = next(data_iter)
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss * -1.0  # to forget instead of learning these examples
        if torch.abs(loss).item() > 1e6:
            print("Loss is too large, breaking after {} iterations".format(i))
            break
        wandb.log({"loss": loss.item(),
                   "log_p": log_p.item(),
                   "log_det": log_det.item(),
                   "prob": log_p.item() + log_det.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f};")

        if (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model, optimizer)


def main():
    args = get_args(forget=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_name = make_exp_dir(args.exp_name)
    print(args)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    wandb.init(project="Glow", entity="malnick", name=args.exp_name, config=args)
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.forget_baseline:
        forget_baseline(args, model, device, forget_optimizer)
    else:
        regular_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        all_transforms: List[Callable] = [transforms.Resize((args.img_size, args.img_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          lambda img: quantize_image(img, args.n_bits)]
        regular_transform = transforms.Compose(all_transforms)
        if args.forget_noise is not None and args.forget_noise > 0:
            additional_transform = lambda img: torch.clamp(img + torch.randn_like(img) * args.forget_noise, -0.5, 0.5)
        else:
            additional_transform = None

        regular_iter: Iterator = iter(sample_data(args.path, args.batch, args.img_size, data_split=args.data_split,
                                                  transform=regular_transform))
        forget_iter: Iterator = get_forget_dataloader(args, additional_tranform=additional_transform)
        forget(args, regular_iter, forget_iter, model, device, forget_optimizer, regular_optimizer)


def evaluate_model(args=None, save_dir='outputs/forget_bpd', save=True, evaluate_celeba=False, dl=None, **kwargs):
    if args is None:
        args = get_args(forget=True)
    file_name = 'bpd.json'
    if save:
        save_dir = os.path.join(save_dir, "bpd")
        if os.path.exists(f'{save_dir}/bpd.json'):
            print("Already evaluated Given folder")
            file_name = 'bpd_3.json'
        os.makedirs(save_dir, exist_ok=True)
        save_dict_as_json(args, f'{save_dir}/args.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.DataParallel = load_model(args, device, training=False)
    n_bins = 2 ** args.n_bits

    assert args.forget_path and args.forget_compare_path
    data = {}
    if dl is not None:
        name = kwargs.get('name', 'dl')
        # evaluating only given dataloader
        cur_bpd = compute_bpd(n_bins, args.img_size, model, device, dl)
        data[name] = cur_bpd
        print(f"{name} BPD: {cur_bpd:.5f}")
        return data
    if evaluate_celeba:
        total_trans = Compose([Resize((args.img_size, args.img_size)), RandomHorizontalFlip(), ToTensor(),
                               lambda img: quantize_image(img, args.n_bits)])
        total_ds = vision_datasets.CelebA(args.path, args.data_split, transform=total_trans, download=False,
                                          target_type='identity')
        total_dl = DataLoader(total_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        total_bpd = compute_bpd(n_bins, args.img_size, model, device, total_dl)
        data['total_bpd'] = total_bpd
        print(f"Total BPD: {total_bpd:.5f}")

    forget_ds = get_forget_dataset(args, images_path=args.forget_path)
    forget_dl = DataLoader(forget_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    forget_bpd = compute_bpd(n_bins, args.img_size, model, device, forget_dl)
    print(f'Forget BPD: {forget_bpd:.5f}')
    data['forget'] = forget_bpd

    # evaluating against all compared images
    compare_ds = get_forget_dataset(args, images_path=args.forget_compare_path, max_size=True)
    compare_dl = DataLoader(compare_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    compare_bpd = compute_bpd(n_bins, args.img_size, model, device, compare_dl)
    print(f'Compare BPD: {compare_bpd:.5f}')

    data['compare'] = compare_bpd
    if save:
        save_dict_as_json(data, f'{save_dir}/{file_name}')
    return data


def evaluate_all_models(glob_pattern="experiments/forget*", evaluate_celeba=False, **kwargs):
    compare_path = "/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/train_set_frequent_identities/4"
    folders = glob(glob_pattern)
    for folder in folders:
        cur_checkpoint = sorted(glob(f'{folder}/checkpoints/model_*.pt'))[-1]
        with open(f'{folder}/args.json') as f:
            cur_base_args = json.load(f)
            cur_base_args = EasyDict(cur_base_args)
        cur_base_args.forget_compare_path = compare_path
        cur_base_args.ckpt_path = cur_checkpoint
        print("Starting folder {}".format(folder))
        evaluate_model(cur_base_args, folder, evaluate_celeba=evaluate_celeba, **kwargs)


def evaluate_base_model():
    args = get_args()
    args.forget_compare_path = "/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/train_set_frequent_identities/4"
    args.forget_path = "/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/train_set_frequent_identities/1"
    data = {}
    cur = time()
    for size in [1, 5, 10, 20, 29]:
        args.forget_size = size
        evaluate_celeba = size == 29
        cur_dict = evaluate_model(args, save=False, evaluate_celeba=evaluate_celeba)
        print(f"Evaluating size {size} took {time() - cur}")
        cur = time()
        data[f'size_{size}'] = cur_dict
    save_dict_as_json(data, "outputs/forget_bpd/base_model_bpd.json")


def evaluate_models_on_celeba(glob_pattern, split='all', img_size=128, n_bits=5):
    # load celeba
    celeba_path = '/home/yandex/AMNLP2021/malnick/datasets/celebA'
    transform = Compose([Resize((img_size, img_size)), RandomHorizontalFlip(), ToTensor(),
                         lambda img: quantize_image(img, n_bits)])
    cur = time()
    ds = vision_datasets.CelebA(celeba_path, split, transform=transform, download=False, target_type='identity')
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=16)
    print("Loaded celeba in ", time() - cur, " seconds")
    evaluate_all_models(glob_pattern, evaluate_celeba=False, dl=dl, name='celebA')


if __name__ == '__main__':
    # evaluate_all_models(glob_pattern="experiments/forget*longer*", evaluate_celeba=True)
    # evaluate_models_on_celeba("experiments/forget_all_images_longer")
    paths = glob("experiments/forget*longer*/bpd/bpd_3.json")
    names = [re.match(r".*/(.*)/.*/.*", p).group(1) for p in paths]
    gather_jsons(paths, names, "outputs/forget_bpd/bpd_3.json")
    # evaluate_base_model()
    # folder = 'experiments/forget_20_images_longer'
    # with open(f"{folder}/args.json") as f:
    #     args = json.load(f)
    #     args = EasyDict(args)
    # args.ckpt_path = '/home/yandex/AMNLP2021/malnick/glow_repos/glow-pytorch-rosinality/experiments/forget_20_images_longer/checkpoints/model_000360.pt'
    # args.forget_compare_path = "/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/train_set_frequent_identities/4"
    # evaluate_model(args, save_dir=folder)
    # main()

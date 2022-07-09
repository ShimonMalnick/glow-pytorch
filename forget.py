import os

from model import Glow
import torch
from utils import get_args, load_model, quantize_image, sample_data, make_exp_dir, save_dict_as_json, \
    save_model_optimizer, CELEBA_ROOT
from torch.utils.data import DataLoader
from typing import Iterator, List, Callable, Dict
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from train import calc_loss
from datasets import CelebAPartial, ForgetSampler
import wandb

FORGET_THRESH = 1e5


def get_partial_dataset(transform, **kwargs) -> CelebAPartial:
    """
    Used for datasets when performing censoring/forgetting. this function is used both to obtain a dataset containing
    only some identities/images, or to obtain a dataset containing all iamges in celeba apart from certain
    identities/images. See documnetaion of CelebAPArtial for more details.
    """
    ds = CelebAPartial(root=CELEBA_ROOT, transform=transform, target_type="identity", **kwargs)
    return ds


def get_data_iterator(ds: CelebAPartial, batch_size, num_workers=16) -> Iterator:
    sampler = None
    shuffle = True
    if len(ds) < batch_size:
        sampler = ForgetSampler(ds, batch_size)
        shuffle = None
    dl_iter = iter(DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))

    while True:
        try:
            yield next(dl_iter)

        except StopIteration:
            dl_iter = iter(
                DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))
            yield next(dl_iter)


def forget(args, remember_iter: Iterator, forget_iter: Iterator, model: Glow, device, forget_optimizer,
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
            cur_batch, _ = next(remember_iter)
            optimizer = regular_optimizer
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss_sign * loss
        if torch.abs(loss).item() > FORGET_THRESH:
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


def args2dset_params(args, ds_type) -> Dict:
    """
    Returns a dictionary of parameters to be passed to the dataset constructor.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: dictionary contating the parameters to be passed to the dataset constructor
    """
    assert ds_type in ['forget', 'remember'], "ds_type must be one of 'forget' or 'remember'"
    out = {"include_only_identities": None,
           "exclude_identities": None,
           "include_only_images": None,
           "exclude_images": None}
    if ds_type == 'forget':
        if args.forget_identity:
            out["include_only_identities"] = [args.forget_identity]
        if args.forget_images:
            out["include_only_images"] = os.listdir(args.forget_images)
    if ds_type == 'remember':
        if args.forget_identity:
            out["exclude_identities"] = [args.forget_identity]
        if args.forget_images:
            out["exclude_images"] = os.listdir(args.forget_images)

    return out


def args2data_iter(args, ds_type, transform) -> Iterator:
    """
    Returns a data iterator for the dataset specified by ds_type.
    :param transform: transform to be applied to the dataset.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: data iterator for the dataset specified by ds_type
    """
    ds_params = args2dset_params(args, ds_type)
    ds = get_partial_dataset(transform=transform, **ds_params)
    return get_data_iterator(ds, args.batch_size)


def forget_baseline(forget_iter: Iterator, args, model, device, optimizer):
    n_bins = 2 ** args.n_bits
    for i in range(args.iter):
        cur_batch, _ = next(forget_iter)
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss * -1.0  # to forget instead of learning these examples
        if torch.abs(loss).item() > FORGET_THRESH:
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
    wandb.init(project="Glow-Forget", entity="malnick", name=args.exp_name, config=args)
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    transform = Compose([Resize((args.img_size, args.img_size)),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, args.n_bits)])
    forget_iter = args2data_iter(args, ds_type='forget', transform=transform)
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.forget_baseline:
        forget_baseline(forget_iter, args, model, device, forget_optimizer)
    else:
        remember_iter = args2data_iter(args, ds_type='remember', transform=transform)
        remember_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        forget(args, remember_iter, forget_iter, model, device, forget_optimizer, remember_optimizer)


if __name__ == '__main__':
    main()

import math
import os
from math import log
import numpy as np
from easydict import EasyDict
from model import Glow
import torch
import torch.nn.functional as F
from utils import get_args, load_model, save_dict_as_json, \
    save_model_optimizer, get_partial_dataset, compute_dataloader_bpd, get_default_forget_transform
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import StepLR
from typing import Iterator, Dict, Union, List, Tuple, Optional
from datasets import CelebAPartial, ForgetSampler
import wandb
import logging
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def calc_forget_loss(log_p, logdet, image_size, n_bins, weights=None):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    if weights is not None:
        assert weights.shape == loss.shape, "weights and loss must have the same shape, but got weights shape {} " \
                                            "and loss shape {}".format(weights.shape, loss.shape)
        loss = (-(loss * weights) / (log(2) * n_pixel)).sum()  # summing because weights are normalized to sum to 1
    else:
        loss = (-loss / (log(2) * n_pixel)).mean()
    return (
        loss,
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def make_forget_exp_dir(exp_name, exist_ok=False, dir_name="forget") -> str:
    base_path = os.path.join("experiments", dir_name, exp_name)
    os.makedirs(f'{base_path}/checkpoints', exist_ok=exist_ok)
    os.makedirs(f'{base_path}/wandb', exist_ok=exist_ok)
    os.makedirs(f'{base_path}/logs', exist_ok=exist_ok)
    return os.path.join(dir_name, exp_name)


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


def calc_batch_bpd(args, model, batch, reduce=True) -> Union[float, torch.Tensor]:
    """
    Calculates the batch-wise BPD of the model.
    :param reduce: if reduce is true, return a sclar value that is the mean of the batch-wise BPDs, if it is false,
    return the per example contriubtion to the BPD, meaning reduced_bpd = sum(unreduced_bpd) / batch_size
    :param args: arguments relevant to the model's parameters and input image size.
    :param model: model to calculate the BPD with.
    :param batch: batch to calculate the BPD of.
    :return: batch-wise BPD of the model.
    """
    n_bins = 2 ** args.n_bits
    M = args.img_size * args.img_size * 3
    with torch.no_grad():
        log_p, logdet, _ = model(batch)
    if reduce:
        cur_nll = - torch.sum(log_p + logdet.mean()).item() / batch.shape[0]
    else:
        cur_nll = - (log_p + logdet.mean())

    cur_bpd = (cur_nll + (M * math.log(n_bins))) / (math.log(2) * M)

    return cur_bpd


def forget(args, remember_iter: Iterator, forget_iter: Iterator, model: Union[Glow, torch.nn.DataParallel],
           device, forget_optimizer, remember_optimizer,
           eval_batches: List[Tuple[str, torch.Tensor, torch.Tensor, Dict]],
           eval_dl: DataLoader,
           forget_eval_data: Tuple[torch.Tensor, Dict],
           scheduler: Optional[StepLR] = None):
    n_bins = 2 ** args.n_bits
    loss_weights = torch.ones(args.batch, device=device) / args.batch if args.adaptive_loss else None
    weights_cache = {"columns": ["step"] + [f"{i}" for i in range(1, args.batch + 1)], "data": []}
    for i in range(args.iter):
        if (i + 1) % args.forget_every == 0:
            loss_sign = -1.0
            loss_name = "forget"
            cur_batch, _ = next(forget_iter)
            optimizer = forget_optimizer
        else:
            loss_sign = 1.0
            loss_name = "remember"
            cur_batch, _ = next(remember_iter)
            optimizer = remember_optimizer
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)
        logdet = logdet.mean()
        if args.adaptive_loss and loss_name == "forget":
            with torch.no_grad():
                loss_weights = compute_adaptive_weights((log_p + logdet).detach(), device=device, scale=1e6)
                weights_cache["data"].append([i + 1] + loss_weights.view(-1).tolist())
                wandb.log({"forget_weights": wandb.Table(**weights_cache)}, commit=False)
        weights = loss_weights if loss_name == "forget" else None
        loss, log_p, log_det = calc_forget_loss(log_p, logdet, args.img_size, n_bins, weights=weights)
        loss = loss_sign * loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None and loss_name == "forget":
            scheduler.step()
        # cur_ref_bpd = calc_batch_bpd(args, model, ref_batch)
        cur_forget_bpd = validation_step(args, i, model, device, eval_dl, eval_batches, forget_eval_data)

        wandb.log({f"{loss_name}": {"loss": loss.item(),
                                    "bpd": cur_forget_bpd.mean().item(),
                                    "log_p": log_p.item(),
                                    "log_det": log_det.item(),
                                    "prob": log_p.item() + log_det.item()},
                   "batch_type": loss_sign
                   })

        if torch.all(cur_forget_bpd > args.forget_thresh):
            logging.info("breaking after {} iterations".format(i + 1))
            wandb.log({f"achieved_thresh": i + 1})
            break

        logging.info(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; "
        )

        if (i + 1) % args.save_every == 0:
            save_forget_checkpoint(args, forget_optimizer, i, model, remember_optimizer, save_optim=False)
    save_forget_checkpoint(args, forget_optimizer, 0, model, remember_optimizer, last=True, save_optim=False)


def save_forget_checkpoint(args, forget_optimizer, iter_num, model, remember_optimizer, last=False, save_optim=True):
    if last:
        model_id = "last"
    else:
        model_id = str(iter_num + 1).zfill(6)
    torch.save(
        model.state_dict(), f'experiments/{args.exp_name}/checkpoints/model_{model_id}.pt'
    )
    if save_optim:
        torch.save(
            forget_optimizer.state_dict(),
            f'experiments/{args.exp_name}/checkpoints/forget_optim_{model_id}.pt'
        )
        torch.save(
            remember_optimizer.state_dict(),
            f'experiments/{args.exp_name}/checkpoints/regular_optim_{model_id}.pt'
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
    if 'data_split' in args and args.data_split in ['train', 'all']:
        out['split'] = args.data_split
    if ds_type == 'forget':
        if args.forget_identity:
            out["include_only_identities"] = [args.forget_identity]
        elif args.forget_images:
            assert args.forget_size, "must specify forget_size if forget_images is specified"
            out["include_only_images"] = os.listdir(args.forget_images)[:args.forget_size]
    if ds_type == 'remember':
        if args.forget_identity:
            out["exclude_identities"] = [args.forget_identity]
        elif args.forget_images:
            assert args.forget_size, "must specify forget_size if forget_images is specified"
            out["exclude_images"] = os.listdir(args.forget_images)[:args.forget_size]

    return out


def args2dataset(args, ds_type, transform):
    ds_params = args2dset_params(args, ds_type)
    ds = get_partial_dataset(transform=transform, **ds_params)
    return ds


def args2data_iter(args, ds_type, transform) -> Iterator:
    """
    Returns a data iterator for the dataset specified by ds_type.
    :param ds_len:
    :param transform: transform to be applied to the dataset.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: data iterator for the dataset specified by ds_type
    """
    ds = args2dataset(args, ds_type, transform)
    return get_data_iterator(ds, args.batch, num_workers=args.num_workers)


def plot_bpd_histograms(step, exp_name, forget_bpd: np.array, eval_bpd: np.array):
    plt.figure()
    plt.hist(forget_bpd, bins=100)
    forget_path = f"experiments/{exp_name}/logs/forget_bpd_hist_{step}.png"
    plt.savefig(forget_path)
    plt.close()

    plt.figure()
    plt.hist(eval_bpd, bins=100)
    eval_path = f"experiments/{exp_name}/logs/eval_bpd_hist_{step}.png"
    plt.savefig(eval_path)
    plt.close()

    plt.figure()
    min_val = min(eval_bpd.min().item(), forget_bpd.min().item())
    max_val = max(eval_bpd.max().item(), forget_bpd.max().item())
    bins = np.linspace(min_val, max_val, 100)
    plt.hist(forget_bpd, bins=bins, label="forget", density=True)
    plt.hist(eval_bpd, bins=bins, label="eval", density=True, alpha=0.5)
    plt.legend()
    both_path = f"experiments/{exp_name}/logs/bpd_hist_{step}.png"
    plt.savefig(both_path)
    plt.close()

    wandb.log({"forget_hist": wandb.Image(forget_path),
               "eval_hist": wandb.Image(eval_path),
               "both_hist": wandb.Image(both_path)}, commit=False)


def validation_step(args: EasyDict, step: int, model: torch.nn.Module, device, dl: DataLoader,
                    eval_batches: List[Tuple[str, torch.Tensor, torch.Tensor, Dict]],
                    forget_data: Tuple[torch.Tensor, Dict]) -> torch.Tensor:
    model.eval()
    forget_batch, forget_dict = forget_data
    forget_bpd = calc_batch_bpd(args, model, forget_batch, reduce=False).cpu()
    forget_data = forget_bpd.view(-1).tolist()
    forget_dict["data"].append([step + 1] + forget_data)
    wandb.log({"forget_bpd": wandb.Table(**forget_dict)}, commit=False)
    if (step + 1) % args.log_every == 0:
        eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model, device, dl, reduce=False).cpu()
        plot_bpd_histograms(step + 1, args.exp_name, forget_bpd.numpy(), eval_bpd.numpy())
        for i, (name, batch, indices, data_dict) in enumerate(eval_batches):
            batch_bpd = calc_batch_bpd(args, model, batch, reduce=False)
            data_dict["data"].append([step + 1] + batch_bpd.view(-1).cpu().tolist())
            wandb.log({name: wandb.Table(**data_dict)}, commit=False)

        wandb.log({f"eval_bpd": eval_bpd.mean().item()},
                  commit=False)
    model.train()

    return forget_bpd


def compute_adaptive_weights(prob, device, scale=1.0) -> torch.Tensor:
    with torch.no_grad():
        weights = F.softmax(scale / prob, dim=0)
        return weights.to(device)


def forget_baseline(forget_iter: Iterator, args, model, device, optimizer,
                    eval_batches: List[Tuple[str, torch.Tensor, torch.Tensor, Dict]],
                    eval_dl: DataLoader,
                    forget_eval_data: Tuple[torch.Tensor, Dict]):
    n_bins = 2 ** args.n_bits
    loss_weights = torch.ones(args.batch, device=device) / args.batch if args.adaptive_loss else None
    weights_cache = {"columns": [f"{i}" for i in range(1, args.batch + 1)], "data": []}
    for i in range(args.iter):
        cur_batch, _ = next(forget_iter)
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)
        logdet = logdet.mean()
        if args.adaptive_loss:
            loss_weights = compute_adaptive_weights((log_p + logdet).detach(), device=device, scale=1e6)
            weights_cache["data"].append(loss_weights.detach().tolist())
            wandb.log({"forget_weights": wandb.Table(**weights_cache)}, commit=False)

        loss, log_p, log_det = calc_forget_loss(log_p, logdet, args.img_size, n_bins, weights=loss_weights)
        loss = loss * -1.0  # to forget instead of learning these examples

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        forget_bpd = validation_step(args, i, model, device, eval_dl, eval_batches, forget_eval_data)

        wandb.log({"loss": loss.item(),
                   "log_p": log_p.item(),
                   "log_det": log_det.item(),
                   "prob": log_p.item() + log_det.item()})

        if torch.all(forget_bpd > args.forget_thresh):
            logging.info("breaking after {} iterations".format(i + 1))
            wandb.log({f"achieved_thresh": i + 1})
            break

        logging.info(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f};")

        if (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model, optimizer, save_prefix='experiments', save_optim=False)
    save_model_optimizer(args, 0, model, optimizer, save_prefix='experiments', last=True, save_optim=False)


def get_random_batch(ds: Dataset, batch_size, device) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(len(ds))[:batch_size]
    batch = [ds[idx][0] for idx in indices]
    batch = torch.stack(batch).to(device)
    return batch, indices


def main():
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="forget")
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)
    forget_iter = get_data_iterator(forget_ds, args.batch, args.num_workers)
    forget_ref_batch = torch.stack([forget_ds[idx][0] for idx in range(len(forget_ds))]).to(device)
    forget_ref_data = (forget_ref_batch, {"columns": ["step"] + [f"idx {i}" for i in range(len(forget_ds))],
                                          "data": []})
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.forget_lr)
    remember_ds = args2dataset(args, ds_type='remember', transform=transform)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    eval_batches = []
    for i in range(1, args.num_ref_batches + 1):
        name = f"ref_batch_{i}"
        ref_batch, indices = get_random_batch(remember_ds, args.batch, device)
        data_dict = {"columns": ['step'] + [f"idx {idx}" for idx in range(args.batch)],
                     "data": []}
        eval_batches.append((name, ref_batch, indices, data_dict))
    ds_rand_perm = torch.randperm(len(remember_ds))
    subset_remember_ds = Subset(remember_ds, ds_rand_perm[:1024])
    eval_dl = DataLoader(subset_remember_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    wandb.init(project="Forget Logged", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    if args.forget_baseline:
        forget_baseline(forget_iter, args, model, device, forget_optimizer, eval_batches, eval_dl, forget_ref_data)
    else:
        remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
        remember_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.scheduler_step and args.scheduler_gamma:
            scheduler = StepLR(optimizer=forget_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        else:
            scheduler = None
        forget(args, remember_iter, forget_iter, model, device, forget_optimizer, remember_optimizer,
               eval_batches, eval_dl, forget_ref_data, scheduler)


if __name__ == '__main__':
    main()

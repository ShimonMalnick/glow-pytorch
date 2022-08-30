import math
import os
from math import log
import numpy as np
from easydict import EasyDict
from model import Glow
import torch
import torch.nn.functional as F
from utils import get_args, load_model, save_dict_as_json, save_model_optimizer, compute_dataloader_bpd,\
    get_default_forget_transform, kl_div_univariate_gaussian, args2dataset, get_interpolated_alpha
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import StepLR
from typing import Iterator, Dict, Union, List, Tuple, Optional
from datasets import CelebAPartial, ForgetSampler
from evaluate import full_experiment_evaluation
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
        log_p, logdet, _ = model(batch + torch.rand_like(batch) / n_bins)
    if reduce:
        cur_nll = - torch.sum(log_p + logdet.mean()).item()
    else:
        cur_nll = - (log_p + logdet.mean())

    cur_bpd = (cur_nll + (M * math.log(n_bins))) / (math.log(2) * M)
    if reduce:
        cur_bpd /= batch.shape[0]
    return cur_bpd


def prob2bpd(prob, n_bins, n_pixels):
    return - prob / (math.log(2) * n_pixels) + math.log(n_bins) / math.log(2)


def compute_relevant_indices(mu, std, thresh, model, forget_batch, n_bins, n_pixels) -> torch.Tensor:
    with torch.no_grad():
        log_p, logdet, _ = model(forget_batch + torch.rand_like(forget_batch) / n_bins)
        logdet = logdet.mean()
        cur_bpd = prob2bpd(log_p + logdet, n_bins, n_pixels)
        indices = torch.nonzero(cur_bpd < (mu + std * thresh), as_tuple=True)[0]
        return indices


def get_log_p_parameters(n_bins, n_pixel, dist, device=None):
    val = -log(n_bins) * n_pixel
    val += dist
    val = -val / (log(2) * n_pixel)

    mean, std = torch.mean(val), torch.std(val)
    if device is not None:
        return mean.to(device), std.to(device)
    return mean, std


def forget_alpha(args, remember_iter: Iterator, forget_iter: Iterator, model: Union[Glow, torch.nn.DataParallel],
                 original_model: Glow,
                 training_devices: List[int],
                 original_model_device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 eval_batches: List[Tuple[str, torch.Tensor, torch.Tensor, Dict]],
                 eval_dl: DataLoader,
                 forget_eval_data: Tuple[torch.Tensor, Dict],
                 scheduler: Optional[StepLR] = None):
    main_device = torch.device(f"cuda:{training_devices[0]}")
    n_bins = 2 ** args.n_bits
    n_pixels = args.img_size * args.img_size * 3
    loss_weights = torch.ones(args.batch, device=main_device) / args.batch if args.adaptive_loss else None
    weights_cache = {"columns": ["step"] + [f"{i}" for i in range(1, args.batch + 1)], "data": []}
    for i in range(args.iter):
        forget_batch = next(forget_iter)[0].to(main_device)
        indices = compute_relevant_indices(args.eval_mu, args.eval_std, args.forget_thresh, model,
                                           forget_batch, n_bins, n_pixels)
        if indices.numel() == 0:
            num_iterations = math.ceil(args.forget_size / args.batch)
            logging.info("A batch above threshold after {} iterations 1/{}".format(i, num_iterations))
            count = 1
            for idx in range(num_iterations - 1):
                forget_batch = next(forget_iter)[0].to(main_device)
                indices = compute_relevant_indices(args.eval_mu, args.eval_std, args.forget_thresh, model,
                                                   forget_batch, n_bins, n_pixels)
                if indices.numel() != 0:
                    break
                logging.info("A batch above threshold after {} iterations {}/{}".format(i, idx + 2, num_iterations))
                count += 1
            if count == num_iterations:
                logging.info("breaking after {} iterations".format(i))
                wandb.log({f"achieved_thresh": i})
                break
        forget_batch = forget_batch[indices]
        wandb.log({"batch size": forget_batch.shape[0]}, commit=False)
        forget_p, forget_det, _ = model(forget_batch + torch.rand_like(forget_batch) / n_bins)
        forget_det = forget_det.mean()
        if args.adaptive_loss:
            with torch.no_grad():
                loss_weights = compute_adaptive_weights((forget_p + forget_det).detach(), device=main_device, scale=1e6)
                weights_cache["data"].append(
                    [i + 1] + loss_weights.view(-1).tolist() + [-1 for _ in range(args.batch - loss_weights.shape[0])])
                wandb.log({"forget_weights": wandb.Table(**weights_cache)}, commit=False)
        forget_loss, forget_p, forget_det = calc_forget_loss(forget_p, forget_det, args.img_size, n_bins,
                                                             weights=loss_weights)
        forget_loss.mul_(-1.0)

        remember_batch = next(remember_iter)[0].to(main_device)
        remember_batch += torch.rand_like(remember_batch) / n_bins
        with torch.no_grad():
            orig_p, orig_det, _ = original_model(remember_batch.to(original_model_device))
            orig_dist = orig_p + orig_det.mean()
            # orig_mean, orig_std = torch.mean(orig_dist).to(main_device), torch.std(orig_dist).to(main_device)
            orig_mean, orig_std = get_log_p_parameters(n_bins, n_pixels, orig_dist, device=main_device)

        remember_p, remember_det, _ = model(remember_batch)
        remember_det = remember_det.mean()
        regular_loss, regular_p, regular_det = calc_forget_loss(remember_p, remember_det, args.img_size, n_bins,
                                                                weights=None)
        remember_dist = remember_p + remember_det
        remember_mean, remember_std = get_log_p_parameters(n_bins, n_pixels, remember_dist)
        kl_loss = kl_div_univariate_gaussian(orig_mean, orig_std, remember_mean, remember_std)
        remember_loss = args.gamma * kl_loss + (1 - args.gamma) * regular_loss

        loss = args.alpha * forget_loss + (1 - args.alpha) * remember_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_forget_bpd = compute_step_stats(args, i, model, main_device, eval_dl, eval_batches, forget_eval_data)

        wandb.log({"forget": {"loss": forget_loss.item()},
                   "remember": {"loss": remember_loss.item()},
                   "forget_bpd_mean": cur_forget_bpd.mean().item()
                   })

        logging.info(f"Iter: {i + 1} Forget Loss: {forget_loss.item():.5f}; Remember Loss: {remember_loss.item():.5f}")

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model.module, optimizer, save_optim=False)
    if args.save_every is not None:
        save_model_optimizer(args, 0, model.module, optimizer, last=True, save_optim=False)

    return model


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
        cur_forget_bpd = compute_step_stats(args, i, model, device, eval_dl, eval_batches, forget_eval_data)

        wandb.log({f"{loss_name}": {"loss": loss.item(),
                                    "log_p": log_p.item(),
                                    "log_det": log_det.item(),
                                    "prob": log_p.item() + log_det.item()},
                   "forget bpd mean": cur_forget_bpd.mean().item()
                   })

        if torch.all(cur_forget_bpd > args.forget_thresh):
            logging.info("breaking after {} iterations".format(i + 1))
            wandb.log({f"achieved_thresh": i + 1})
            break

        logging.info(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; "
        )

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_forget_checkpoint(args, forget_optimizer, i, model, remember_optimizer, save_optim=False)
    if args.save_every is not None:
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


def plot_bpd_histograms(step, exp_name, forget_bpd: Optional[np.array] = None, eval_bpd: Optional[np.array] = None):
    log_params = {}
    if forget_bpd is not None:
        plt.figure()
        plt.hist(forget_bpd, bins=100)
        forget_path = f"experiments/{exp_name}/logs/forget_bpd_hist_{step}.png"
        plt.savefig(forget_path)
        plt.close()
        log_params["forget_hist"] = wandb.Image(forget_path)

    if eval_bpd is not None:
        plt.figure()
        plt.hist(eval_bpd, bins=100)
        eval_path = f"experiments/{exp_name}/logs/eval_bpd_hist_{step}.png"
        plt.savefig(eval_path)
        plt.close()
        log_params["eval_hist"] = wandb.Image(eval_path)

    if eval_bpd is not None and forget_bpd is not None:
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
        log_params["both_hist"] = wandb.Image(both_path)

    wandb.log(log_params, commit=False)


def compute_step_stats(args: EasyDict, step: int, model: torch.nn.Module, device, dl: DataLoader,
                       eval_batches: List[Tuple[str, torch.Tensor, torch.Tensor, Dict]],
                       forget_data: Tuple[torch.Tensor, Dict]) -> torch.Tensor:
    model.eval()
    forget_batch, forget_dict = forget_data
    forget_bpd = calc_batch_bpd(args, model, forget_batch, reduce=False).cpu()
    forget_data = forget_bpd.view(-1).tolist()
    forget_dict["data"].append([step] + forget_data)
    wandb.log({"forget_bpd": wandb.Table(**forget_dict)}, commit=False)
    if (step + 1) % args.log_every == 0:
        eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model, device, dl, reduce=False).cpu()
        args.eval_mu = eval_bpd.mean().item()
        args.eval_std = eval_bpd.std().item()
        logging.info(f"eval_mu: {args.eval_mu}, eval_std: {args.eval_std} for iteration {step}")
        plot_bpd_histograms(step, args.exp_name, forget_bpd.numpy(), eval_bpd.numpy())
        for i, (name, batch, indices, data_dict) in enumerate(eval_batches):
            batch_bpd = calc_batch_bpd(args, model, batch, reduce=False)
            data_dict["data"].append([step] + batch_bpd.view(-1).cpu().tolist())
            wandb.log({name: wandb.Table(**data_dict)}, commit=False)

        wandb.log({f"eval_bpd": eval_bpd.mean().item(),
                   "eval_mu": args.eval_mu,
                   "eval_std": args.eval_std},
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

        forget_bpd = compute_step_stats(args, i + 1, model, device, eval_dl, eval_batches, forget_eval_data)

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


def get_random_batch(ds: Dataset, batch_size, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(len(ds))[:batch_size]
    batch = [ds[idx][0] for idx in indices]
    batch = torch.stack(batch).to(device)
    return batch, indices


def get_eval_data(args, remember_ds, device=None) -> Tuple[
    List[Tuple[str, torch.Tensor, torch.Tensor, Dict]], DataLoader]:
    eval_batches = []
    if args.num_ref_batches is not None:
        for i in range(1, args.num_ref_batches + 1):
            name = f"ref_batch_{i}"
            ref_batch, indices = get_random_batch(remember_ds, args.batch, device=device)
            data_dict = {"columns": ['step'] + [f"idx {idx}" for idx in range(args.batch)],
                         "data": []}
            eval_batches.append((name, ref_batch, indices, data_dict))
    ds_rand_perm = torch.randperm(len(remember_ds))
    subset_remember_ds = Subset(remember_ds, ds_rand_perm[:1024])
    eval_dl = DataLoader(subset_remember_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
    return eval_batches, eval_dl


def main():
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1]
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="forget_kl_paper")
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices,
                                              output_device=train_devices[0])
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)
    forget_iter = get_data_iterator(forget_ds, args.batch, args.num_workers)
    forget_ref_batch = torch.stack([forget_ds[idx][0] for idx in range(len(forget_ds))])
    forget_ref_data = (forget_ref_batch, {"columns": ["step"] + [f"idx {i}" for i in range(len(forget_ds))],
                                          "data": []})
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.forget_lr)
    remember_ds = args2dataset(args, ds_type='remember', transform=transform)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    eval_batches, eval_dl = get_eval_data(args, remember_ds, device=None)
    args.alpha = get_interpolated_alpha(args.forget_size)
    wandb.init(project="Forget-KL-paper", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    if args.forget_baseline:
        forget_baseline(forget_iter, args, model, None, forget_optimizer, eval_batches, eval_dl, forget_ref_data)
    else:
        compute_step_stats(args, 0, model, None, eval_dl, eval_batches, forget_ref_data)
        remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
        if args.scheduler_step and args.scheduler_gamma:
            scheduler = StepLR(optimizer=forget_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        else:
            scheduler = None

        logging.info("Starting forget alpha procedure")
        finetuned_model = forget_alpha(args, remember_iter, forget_iter, model,
                     original_model, train_devices, original_model_device,
                     forget_optimizer, eval_batches, eval_dl,
                     forget_ref_data, scheduler=scheduler)
        full_experiment_evaluation(f"experiments/{args.exp_name}", args, partial=10000, model=finetuned_model)

if __name__ == '__main__':
    main()

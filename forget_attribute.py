from typing import Iterator, Union, List
import wandb
from torch.utils.data import Subset, Dataset, DataLoader
from easydict import EasyDict
from attribute_classifier import CelebaAttributeCls, DEFAULT_CLS_CKPT, load_indices_cache
import torch
from forget import make_forget_exp_dir, full_experiment_evaluation, get_data_iterator, get_kl_loss_fn, \
    get_forget_distance_loss, get_log_p_parameters, get_kl_and_remember_loss, prob2bpd
from utils import set_all_seeds, get_args, get_default_forget_transform, load_model, CELEBA_ROOT, \
    save_dict_as_json, compute_dataloader_bpd, save_model_optimizer
import logging
from model import Glow
from torchvision.datasets import CelebA


def load_cls(ckpt_path: str = DEFAULT_CLS_CKPT, device=None) -> CelebaAttributeCls:
    model = CelebaAttributeCls.load_from_checkpoint(ckpt_path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def compute_attribute_step_stats(args: EasyDict, step: int, model: torch.nn.Module, device, remember_ds: Dataset,
                                 forget_ds: Dataset) -> torch.Tensor:
    model.eval()
    cur_forget_indices = torch.randperm(len(forget_ds))[:1024]
    cur_forget_ds = Subset(forget_ds, cur_forget_indices)
    cur_forget_dl = DataLoader(cur_forget_ds, batch_size=256, num_workers=args.num_workers)

    cur_remember_indices = torch.randperm(len(remember_ds))[:1024]
    cur_remember_ds = Subset(remember_ds, cur_remember_indices)
    cur_remember_dl = DataLoader(cur_remember_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    if (step + 1) % args.log_every == 0:
        eval_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model,
                                          device, cur_remember_dl, reduce=False).cpu()
        args.eval_mu = eval_bpd.mean().item()
        args.eval_std = eval_bpd.std().item()

        forget_bpd = compute_dataloader_bpd(2 ** args.n_bits, args.img_size, model,
                                            device, cur_forget_dl, reduce=False).cpu()

        logging.info(
            f"eval_mu: {args.eval_mu}, eval_std: {args.eval_std}, forget_bpd: {forget_bpd.mean().item()} for iteration {step}")
        forget_signed_distance = (forget_bpd.mean().item() - args.eval_mu) / args.eval_std
        wandb.log({f"eval_bpd": eval_bpd.mean().item(),
                   "eval_mu": args.eval_mu,
                   "eval_std": args.eval_std,
                   "forget_distance": forget_signed_distance},
                  commit=False)
        return forget_bpd

    return torch.tensor([0], dtype=torch.float)


def forget_attribute(args: EasyDict, remember_ds: Dataset, forget_ds: Dataset,
                     model: Union[Glow, torch.nn.DataParallel],
                     original_model: Glow,
                     training_devices: List[int],
                     original_model_device: torch.device,
                     optimizer: torch.optim.Optimizer):
    kl_loss_fn = get_kl_loss_fn(args.loss)
    sigmoid = torch.nn.Sigmoid()
    main_device = torch.device(f"cuda:{training_devices[0]}")
    n_bins = 2 ** args.n_bits
    n_pixels = args.img_size * args.img_size * 3

    remember_iter = get_data_iterator(remember_ds, args.batch, args.num_workers)
    forget_iter = get_data_iterator(forget_ds, args.batch, args.num_workers)
    for i in range(args.iter):
        cur_forget_images = next(forget_iter)[0].to(main_device)
        log_p, logdet, _ = model(cur_forget_images + torch.rand_like(cur_forget_images) / n_bins)
        logdet = logdet.mean()
        cur_bpd = prob2bpd(log_p + logdet, n_bins, n_pixels)
        signed_distance = (cur_bpd - (args.eval_mu + args.eval_std * (args.forget_thresh + 0.3)))
        forget_loss = sigmoid(signed_distance ** 2).mean()

        remember_batch = next(remember_iter)[0].to(main_device)
        remember_batch += torch.rand_like(remember_batch) / n_bins
        with torch.no_grad():
            orig_p, orig_det, _ = original_model(remember_batch.to(original_model_device))
            orig_dist = orig_p + orig_det.mean()
            orig_mean, orig_std = get_log_p_parameters(n_bins, n_pixels, orig_dist, device=main_device)

        kl_loss, remember_loss = get_kl_and_remember_loss(args, kl_loss_fn, model, n_bins, n_pixels, orig_mean,
                                                          orig_std, remember_batch)

        loss = args.alpha * forget_loss + (1 - args.alpha) * remember_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_forget_bpd = compute_attribute_step_stats(args, i, model, main_device, remember_ds, forget_ds)
        wandb.log({"forget": {"loss": forget_loss.item(), "kl_loss": kl_loss.item()},
                   "remember": {"loss": remember_loss.item()},
                   "forget_bpd_mean": cur_forget_bpd.mean().item()
                   })

        logging.info(
            f"Iter: {i + 1} Forget Loss: {forget_loss.item():.5f}; Remember Loss: {remember_loss.item():.5f}")

        if args.save_every is not None and (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model.module, optimizer, save_optim=False)
    if args.save_every is not None:
        save_model_optimizer(args, 0, model.module, optimizer, last=True, save_optim=False)

    return model


def main():
    set_all_seeds(seed=37)
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True, forget_attribute=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1]
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="forget_attributes")
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices,
                                              output_device=train_devices[0])
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)

    assert args.forget_attribute is not None, "Must specify attribute to forget"
    forget_indices = load_indices_cache(args.forget_attribute)

    celeba_ds = CelebA(root=CELEBA_ROOT, split='train',
                       transform=transform, target_type='attr')
    forget_ds = Subset(celeba_ds, forget_indices)
    remember_ds = Subset(celeba_ds, list(set(range(len(celeba_ds))) - set(forget_indices.tolist())))

    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    wandb.init(project="forget_attributes", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    compute_attribute_step_stats(args, 0, model, None, remember_ds, forget_ds)
    logging.info("Starting forget attribute procedure")
    forget_attribute(args, remember_ds, forget_ds, model,
                     original_model, train_devices, original_model_device,
                     forget_optimizer)


if __name__ == '__main__':
    main()

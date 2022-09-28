import json
import os
import wandb
from easydict import EasyDict
from torch.utils.data import Subset, Dataset, DataLoader
import torch
from forget import make_forget_exp_dir
from utils import set_all_seeds, get_args, get_default_forget_transform, load_model, save_dict_as_json, \
    compute_dataloader_bpd
import logging
from model import Glow
from fairface_dataset import FairFaceDataset, LOCAL_FAIRFACE_ROOT, one_type_label_wrapper
from forget_attribute import forget_attribute, generate_random_samples
from evaluate import compute_ds_distribution


def compute_group_step_stats(args: EasyDict,
                             step: int,
                             model: torch.nn.DataParallel,
                             device,
                             remember_ds: Dataset,
                             forget_ds: Dataset,
                             sampling_device: torch.device,
                             init=False) -> torch.Tensor:
    model.eval()
    cur_forget_dl = DataLoader(forget_ds, batch_size=256, shuffle=True, num_workers=args.num_workers)

    cur_remember_dl = DataLoader(remember_ds, batch_size=256, shuffle=True, num_workers=args.num_workers)

    if (step + 1) % args.log_every == 0 or init:
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
        # to generate images using the reverse funciton, we need the module itself from the DataParallel wrapper
        generate_random_samples(model.module, sampling_device, args,
                                f"experiments/{args.exp_name}/random_samples/step_{step}.pt")
        return forget_bpd

    return torch.tensor([0], dtype=torch.float)


def compute_group_distribution(exp_dir, **kwargs):
    with open(f"{exp_dir}/args.json", 'r') as f:
        args = EasyDict(json.load(f))
    if 'ckpt_path' in kwargs:
        args.ckpt_path = kwargs['ckpt_path']
    else:
        args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    age_label = 1  # label for ages 3-9
    fairface_ds = FairFaceDataset(root=LOCAL_FAIRFACE_ROOT,
                                  transform=transform,
                                  target_transform=one_type_label_wrapper('age', one_hot=False),
                                  data_type=args.data_split)
    for group in ["forget", "remember"]:
        save_dir = f"{exp_dir}/distribution_stats/{group}"
        indices = torch.load(f"{LOCAL_FAIRFACE_ROOT}/age_indices/{args.data_split}/{group}_indices_{age_label}.pt")
        ds = Subset(fairface_ds, indices)
        if 'suff' in kwargs:
            save_dir += f"_{kwargs['suff']}"
        compute_ds_distribution(256, save_dir, training=False, args=args, device=device, ds=ds, **kwargs)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1]
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="forget_children")
    os.makedirs(f"experiments/{args.exp_name}/random_samples")
    logging.info(args)
    model: torch.nn.DataParallel = load_model(args, training=True, device_ids=train_devices,
                                              output_device=train_devices[0])
    original_model: Glow = load_model(args, device=original_model_device, training=False)
    original_model.requires_grad_(False)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    age_label = 1  # label for ages 3-9
    args['age_label'] = age_label
    split = 'train'
    fairface_ds = FairFaceDataset(root=LOCAL_FAIRFACE_ROOT,
                                  transform=transform,
                                  target_transform=one_type_label_wrapper('age', one_hot=False),
                                  data_type=split)
    load_from_cache = True
    if load_from_cache:
        forget_indices = torch.load(f"{LOCAL_FAIRFACE_ROOT}/age_indices/{split}/forget_indices_{age_label}.pt")
        remember_indices = torch.load(f"{LOCAL_FAIRFACE_ROOT}/age_indices/{split}/remember_indices_{age_label}.pt")
    else:
        assert args.remember_group_size is not None and args.forget_group_size is not None, \
            "remember_group_size and forget_group_size must be specified if not loading from cache"
        assert args.remember_group_size + args.forget_group_size <= len(fairface_ds), \
            f"remember_group_size + forget_group_size must be less than the total number of samples which " \
            f"is {len(fairface_ds)}, but got {args.remember_group_size + args.forget_group_size}"
        rand_indices = torch.randperm(len(fairface_ds))
        forget_indices = rand_indices[:args.forget_group_size]
        remember_indices = rand_indices[args.forget_group_size:args.forget_group_size + args.remember_group_size]
    forget_ds = Subset(fairface_ds, forget_indices)
    remember_ds = Subset(fairface_ds, remember_indices)
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args["remember_ds_len"] = len(remember_ds)
    args["forget_ds_len"] = len(forget_ds)
    wandb.init(project="forget_children", entity="malnick", name=args.exp_name, config=args,
               dir=f'experiments/{args.exp_name}/wandb')
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')

    compute_group_step_stats(args, 0, model, None, remember_ds, forget_ds, train_devices[0], init=True)
    logging.info("Starting forget attribute procedure")
    forget_attribute(args, remember_ds, forget_ds, model,
                     original_model, train_devices, original_model_device,
                     forget_optimizer)


if __name__ == '__main__':
    set_all_seeds(37)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    main()
    # exp = "experiments/forget_children/forget_1000_lr_1e-4"
    # compute_group_distribution(exp)
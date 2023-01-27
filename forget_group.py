import json
import math
import os
import sys
from glob import glob
import numpy as np
import wandb
from easydict import EasyDict
from torch.utils.data import Subset, Dataset, DataLoader
import torch
from forget import make_forget_exp_dir
from utils import set_all_seeds, get_args, get_default_forget_transform, load_model, save_dict_as_json, \
    compute_dataloader_bpd, plotly_init, save_fig, set_fig_config, np_gaussian_pdf
import plotly.graph_objects as go
from plotly.colors import qualitative
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
                             init: bool = False,
                             generate_samples: bool = True) -> torch.Tensor:
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
        if generate_samples:
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
        if group == 'forget' and args.forget_group_size is not None:
            print(f"Using {args.forget_group_size} samples from {group} group")
            indices = indices[:args.forget_group_size]
        if group == 'remember' and args.remember_group_size is not None:
            print(f"Using {args.remember_group_size} samples from {group} group")
            indices = indices[:args.remember_group_size]
        ds = Subset(fairface_ds, indices)
        if 'suff' in kwargs:
            save_dir += f"_{kwargs['suff']}"
        compute_ds_distribution(256, save_dir, training=False, args=args, device=device, ds=ds, **kwargs)


def plot_distributions(exp_dir, skip_reference=False, base=False):
    plotly_init()
    fig = go.Figure()
    fig.update_layout(showlegend=True, plot_bgcolor='rgba(0,0,0,0)')
    distributions = os.listdir(f"{exp_dir}/distribution_stats")
    file_name = "nll_distribution.pt"
    remember_mean, remember_std = None, None
    colors = qualitative.D3
    M = 128 * 128 * 3
    n_bins = 2 ** 5
    for i, d in enumerate(distributions):
        cur_name = d
        cur_color = colors[i]
        if 'partial' in d:
            cur_name = "reference"
            if skip_reference:
                continue
        if d == 'forget':
            continue
        cur_dist = torch.load(f"{exp_dir}/distribution_stats/{d}/{file_name}").numpy()
        cur_dist = (cur_dist + (M * math.log(n_bins))) / (math.log(2) * M)
        n_points = int(math.sqrt(cur_dist.size))
        x = np.linspace(cur_dist.min(), cur_dist.max(), n_points)
        y = np_gaussian_pdf(x, cur_dist.mean(), cur_dist.std())
        if d == 'remember':
            remember_mean, remember_std = cur_dist.mean(), cur_dist.std()
            line_params = dict(color=colors[0])
        else:
            line_params = dict(color=colors[6], dash='dash')
        fig.add_trace(go.Scatter(x=x, y=y, name=cur_name, line=line_params))
    assert remember_mean is not None and remember_std is not None

    forget_x = torch.load(f"{exp_dir}/distribution_stats/forget/{file_name}").numpy()
    forget_x = (forget_x + (M * math.log(n_bins))) / (math.log(2) * M)
    forget_y = np_gaussian_pdf(forget_x, remember_mean, remember_std)
    fig.add_trace(go.Scatter(x=forget_x, y=forget_y, name='forget', mode='markers',
                             marker=dict(size=8, line_width=2, color=colors[1])))

    baseline_dist = torch.load("models/baseline/continue_celeba/distribution_stats/valid_partial_10000/nll_distribution.pt").numpy()
    baseline_dist = (baseline_dist + (M * math.log(n_bins))) / (math.log(2) * M)
    n_points = int(math.sqrt(baseline_dist.size))
    x = np.linspace(baseline_dist.min(), baseline_dist.max(), n_points)
    y = np_gaussian_pdf(x, baseline_dist.mean(), baseline_dist.std())
    if base:
        name = "base"
    else:
        name = "baseline"
    fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=colors[4], dash='dash')))
    set_fig_config(fig, font_size=16)
    fig.update_xaxes(title='BPD')
    fig.update_yaxes(title='Density')
    suff = "" if skip_reference else "_w_ref"
    save_fig(fig, f"{exp_dir}/distribution_plot_{name}{suff}.pdf")


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = get_args(forget=True, forget_group=True)
    all_devices = list(range(torch.cuda.device_count()))
    train_devices = all_devices[:-1]
    original_model_device = torch.device(f"cuda:{all_devices[-1]}")
    args.exp_name = make_forget_exp_dir(args.exp_name, exist_ok=False, dir_name="supp_group")
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
        if args.forget_group_size is not None:
            assert args.forget_group_size <= len(forget_indices),\
                f"forget group size {args.forget_group_size} is too large"
            forget_indices = forget_indices[:args.forget_group_size]
        remember_indices = torch.load(f"{LOCAL_FAIRFACE_ROOT}/age_indices/{split}/remember_indices_{age_label}.pt")
        if args.remember_group_size is not None:
            assert args.remember_group_size <= len(remember_indices),\
                f"remember group size {args.remember_group_size} is too large"
            remember_indices = remember_indices[:args.remember_group_size]
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

    compute_group_step_stats(args, 0, model, None, remember_ds, forget_ds, train_devices[0], init=True,
                             generate_samples=True)
    logging.info("Starting forget attribute procedure")
    forget_attribute(args, remember_ds, forget_ds, model,
                     original_model, train_devices, original_model_device,
                     forget_optimizer, generate_samples=True)


def plot_distances(exp_dirs, prob_dist=True):
    values = []
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/args.json", "r") as f:
            args = EasyDict(json.load(f))
        cur_key = args.forget_ds_len
        remember_dist = torch.load(f"{exp_dir}/distribution_stats/remember/nll_distribution.pt")
        mu, sigma = remember_dist.mean(), remember_dist.std()
        forget_dist = torch.load(f"{exp_dir}/distribution_stats/forget/nll_distribution.pt")
        forget_mean_dist = ((forget_dist - mu) / sigma).mean()
        values.append((cur_key, forget_mean_dist))

    values = sorted(values, key=lambda x: x[0])
    x, y = zip(*values)
    if prob_dist:
        y = 1 - 0.5 * (1 + torch.erf(torch.tensor(y) / np.sqrt(2)))

    plotly_init()
    fig = go.Figure(layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    set_fig_config(fig, font_size=18)
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines+markers'))
    fig.update_xaxes(title='# Images')
    if not prob_dist:
        fig.update_yaxes(title='AMSD')
    else:
        fig.update_yaxes(title='Likelihood quantile')

    fig.update_layout(
        yaxis=dict() if prob_dist else dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['0', '1σ', '2σ', '3σ', '4σ'],
        ),
        xaxis=dict(tickmode='array',
                   tickvals=[i for i in range(0, 180, 20)],
                   ticktext=[str(i) for i in range(0, 180, 20)]))
    fig.write_image("images/no_data_access_probs.pdf")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    set_all_seeds(37)
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    main()
    # exps = glob("experiments/forget_children_long/*")
    # exps = [exp for exp in exps if (not int(exp.split("_")[-1]) > 120) and
    #         os.path.exists(f"{exp}/distribution_stats/remember")]
    # plot_distances(exps)
    # exp = sys.argv[1]
    # exp = "experiments/forget_children/forget_10"
    # plot_distributions(exp, skip_reference=False, base=True)

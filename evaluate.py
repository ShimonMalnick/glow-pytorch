import json
import math
import re
from time import time
import logging
import numpy as np
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, CELEBA_ROOT, \
    compute_dataset_bpd, get_default_forget_transform, np_gaussian_pdf, data_parallel2normal_state_dict
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List, Union
from datasets import CelebAPartial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def eval_model(dset_tuples: Iterable[Tuple[str, Dataset]], n_bits: int, model, device, img_size: int = 128,
               **kwargs) -> dict:
    n_bins = 2 ** n_bits
    data = {}
    start = time()
    for name, dataset in dset_tuples:
        if 'reduce' in kwargs:
            reduce = kwargs['reduce']
            cur_bpd = compute_dataset_bpd(n_bins, img_size, model, device, dataset, reduce=reduce)
            if not reduce:
                data[name] = {'bpd': cur_bpd.mean().item(),
                              'bpd_std': cur_bpd.std().item(),
                              'bpd_min': cur_bpd.min().item(),
                              'bpd_max': cur_bpd.max().item(),
                              'values': cur_bpd.tolist()}
            else:
                data[name] = cur_bpd
        else:
            cur_bpd = compute_dataset_bpd(n_bins, img_size, model, device, dataset)
            data[name] = cur_bpd
        logging.info(f'{name} done in {time() - start}')
        start = time()
    # if 'save_dir' in kwargs:
    #     save_dir = kwargs['save_dir']
    #     save_dict_as_json(data, f'{save_dir}/bpd.json')
    return data


def eval_models(ckpts: Iterable[Tuple[str, str]], args: Dict, dsets_tuples: Iterable[Tuple[str, Dataset]], **kwargs):
    """
    Used for Evaluation of multiple models on the same data sets
    :param dsets_tuples:
    :param dsets: tuples of (name, ds) where ds is a dataset and name is the corresponding name of that data.
    :param ckpts: list of tuples of (ckpt, names) where ckpt is a path to a checkpoint and name is the corresponding
     model name
    :param args: arguments relevant for the models (same arguments to all models)
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    start = time()
    for name, ckpt in ckpts:
        args.ckpt_path = ckpt
        model: Glow = load_model(args, device, training=False)
        model.eval()
        data[name] = eval_model(dsets_tuples, args.n_bits, model, device, img_size=args.img_size, **kwargs)
        print(f'{name} done in {time() - start}')
        start = time()
    if 'save_path' in kwargs:
        save_path = kwargs['save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.isfile(save_path):
            with open(save_path, 'r') as f:
                old_data = json.load(f)
            for key in old_data:
                if key in data:
                    data[key].update(old_data[key])
                else:
                    data[key] = old_data[key]
        save_dict_as_json(data, save_path)

    return data


def run_eval_on_models_on_images(folders: List[str], images_folders: List[str], names: List[str], **kwargs):
    assert folders and images_folders
    args = get_args(forget=True)
    ckpts = [(os.path.basename(os.path.normpath(folder)), f"{folder}/checkpoints/model_last.pt") for folder in folders]
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    dsets = [(name, CelebAPartial(root=CELEBA_ROOT, transform=transform, target_type='identity', split='train',
                                  include_only_images=os.listdir(folder))) for name, folder in
             zip(names, images_folders)]
    return eval_models(ckpts, args, dsets, **kwargs)


def run_eval_on_models(folders: List[str], dsets_identities: List[int], **kwargs):
    """
    Evaluates all models on the same data sets
    :param dsets_identities: identities representing the different data sets. every identity means the models will
    be evaluated on a dataset consisting all the images of the given identity.
    :param folders: folders to evaluate (that include ckpts)
    :param kwargs: arguments relevant for the models (same arguments to all models)
    :return:
    """
    assert folders
    args = get_args(forget=True)
    ckpts: List[Tuple[str, str]] = []
    dsets: List[Tuple[str, Dataset]] = []
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    base_ds = CelebA(root=CELEBA_ROOT, transform=transform, target_type='identity', split='train')
    with open("outputs/celeba_stats/identity2indices.json", 'r') as f:
        identity2indices = json.load(f)
    for identity in dsets_identities:
        if identity > 0:
            cur_ds = Subset(base_ds, identity2indices[str(identity)])
            dsets.append((f'{identity}', cur_ds))
        else:
            dsets.append((f'celeba_train', base_ds))
    for folder in folders:
        last_model_path = f'{folder}/checkpoints/model_last.pt'
        if os.path.isfile(last_model_path):
            cur_checkpoint = last_model_path
        else:
            logging.warning(f'No last checkpoint found in {folder}')
            cur_checkpoint = sorted(glob(f'{folder}/checkpoints/model_*.pt'))[-1]
        ckpts.append((os.path.basename(os.path.normpath(folder)), cur_checkpoint))
    return eval_models(ckpts, args, dsets, **kwargs)


def compute_similarity_vs_bpd(folders: List[str], similarities_json: str = 'outputs/celeba_stats/1_similarities.json',
                              step=1, **kwargs):
    """
    Compute data for a plot of similarity (x-axis) vs bpd (y-axis) of different identities, for the given models that
    reside in folders
    :param step: Step on the number of identities to use for the plot
    :param similarities_json:
    :param folders: folders to evaluate (that include ckpts)
    :param kwargs: arguments relevant for the models (same arguments to all models)
    :return:
    """
    start = time()
    args = get_args(forget=True)
    with open(similarities_json, 'r') as f:
        similarities_dict = json.load(f)
    relevant_ids = [int(k) for k in similarities_dict.keys()]
    relevant_ids = [relevant_ids[i] for i in range(0, len(relevant_ids), step)]
    for folder in folders:
        last_model_path = f'{folder}/checkpoints/model_last.pt'
        if os.path.isfile(last_model_path):
            cur_checkpoint = last_model_path
        else:
            raise ValueError(f'No last checkpoint found in {folder}')
            # cur_checkpoint = sorted(glob(f'{folder}/checkpoints/model_*.pt'))[-1]
        args.ckpt_path = cur_checkpoint
        model: Glow = load_model(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"), training=False)
        model.eval()
        dsets: List[Tuple[str, Dataset]] = []
        transform = get_default_forget_transform(args.img_size, args.n_bits)
        base_ds = CelebA(root=CELEBA_ROOT, transform=transform, target_type='identity', split='train')
        with open("outputs/celeba_stats/identity2indices.json", 'r') as f:
            identity2indices = json.load(f)
        for identity in relevant_ids:
            cur_ds = Subset(base_ds, identity2indices[str(identity)])
            dsets.append((f'identity_{identity}', cur_ds))
        data = eval_models([(os.path.basename(os.path.normpath(folder)), cur_checkpoint)], args, dsets, **kwargs)
        save_path = f'{folder}/bpd_vs_similarity.json'
        save_dict_as_json(data, save_path)
        print(f'{folder} done in {time() - start}')
        start = time()


def plot_similarity_vs_bpd(save_path: str, bpd_json: str,
                           similarity_json: str = 'outputs/celeba_stats/1_similarities.json',
                           outliers_thresh=None, distance=False):
    with open(bpd_json, 'r') as f:
        bpd_dict = json.load(f)
    assert len(bpd_dict) == 1
    bpd_dict = bpd_dict[list(bpd_dict.keys())[0]]
    pattern = r"identity_(\d+)"
    bpd_dict = {re.match(pattern, k).group(1): v for k, v in bpd_dict.items()}
    with open(similarity_json, 'r') as f:
        similarity_dict = json.load(f)
    if outliers_thresh is not None:
        data = [(bpd_dict[k], similarity_dict[k]) for k in bpd_dict.keys() if bpd_dict[k] < outliers_thresh]
    else:
        data = [(bpd_dict[k], similarity_dict[k]) for k in bpd_dict.keys()]
    if distance:
        data = [(d[0], 1 - d[1]) for d in data]  # cosine_distance = 1 - cosine_similarity

    # data should already be sorted but just in case
    data.sort(key=lambda x: x[1])
    x_axis = [x[1] for x in data]
    y_axis = [x[0] for x in data]
    plt.plot(x_axis, y_axis)
    graph_type = 'Distance' if distance else 'Similarity'
    if outliers_thresh is not None:
        plt.title(f"Cosine {graph_type} vs BPD (without outliers above {outliers_thresh})")
    else:
        plt.title(f'Cosine {graph_type} vs BPD of different identities')
    plt.xlabel(f'Cosine {graph_type}')
    plt.ylabel('BPD')
    plt.savefig(save_path)
    plt.close()


def compute_ds_distribution(batch_size, save_dir: str = 'outputs/celeba_stats/bpd_distribution', training=False,
                            args=None, device=None, ds=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args is None:
        args = get_args(forget=True)
    model: Glow = load_model(args, device, training=training)
    scores = None

    if ds is None:
        transform = get_default_forget_transform(args.img_size, args.n_bits)
        ds = CelebA(root=CELEBA_ROOT, target_type='identity', split='train', transform=transform)
    # ds_debug = Subset(ds, range(2048))
    # dl = DataLoader(ds_debug, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

    M = args.img_size * args.img_size * 3
    n_bins = 2 ** args.n_bits
    for i, batch in enumerate(dl, 1):
        x, _ = batch
        x = x.to(device)
        x += torch.rand_like(x) / n_bins
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        cur_nll = - (log_p + logdet)
        assert cur_nll.nelement() == batch_size and cur_nll.ndim == 1
        cur_score = cur_nll
        if scores is None:
            scores = cur_score.detach().cpu()
        else:
            scores = torch.cat((scores, cur_score.detach().cpu()))
        logging.info(f"Finished {i}/{len(dl)} batches")
    bpd = scores / (math.log(2) * M) + math.log(n_bins) / math.log(2)
    torch.save(scores, f"{save_dir}/nll_distribution.pt")
    data = {'nll':
                {'mean': torch.mean(scores).item(),
                 'std': torch.std(scores).item(),
                 'min': torch.min(scores).item(),
                 'max': torch.max(scores).item(),
                 'median': torch.median(scores).item()},
            'bpd':
                {'mean': torch.mean(bpd).item(),
                 'std': torch.std(bpd).item(),
                 'min': torch.min(bpd).item(),
                 'max': torch.max(bpd).item(),
                 'median': torch.median(bpd).item()}}
    save_dict_as_json(data, f"{save_dir}/distribution.json")
    plot_distribution(-1 * scores.numpy(), f"{save_dir}/ll_distribution_hist.png", normal_estimation=True, density=True,
                      title=None, legend=[r'$log(p^{\theta}_X(x))$', r'$\mathcal{N}(\mu, \sigma)$'])
    plot_distribution(bpd.numpy(), f"{save_dir}/bpd_distribution_hist.png", normal_estimation=True, density=True,
                      title=None, legend=[r'$BPD(x)$', r'$\mathcal{N}(\mu, \sigma)$'])


def plot_distribution(tensors: Union[str, np.ndarray], save_path, normal_estimation=True, density=True,
                      title=None, legend=None, n_bins_plot=80):
    if isinstance(tensors, str):
        scores = torch.load(tensors).numpy()
    elif isinstance(tensors, np.ndarray):
        scores = tensors
    else:
        raise ValueError(f"tensors must be either a path to a file or a numpy array")
    import matplotlib as mpl
    from pylab import cm
    import matplotlib.font_manager as fm
    font_names = [f.name for f in fm.fontManager.ttflist]
    # print(font_names)
    # mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    # Generate 2 colors from the 'tab10' colormap
    colors = cm.get_cmap('tab20')
    plt.figure(figsize=(6, 4))
    # Plot histogram
    # legend = [r"$\logP_X^\theta$", r"$\mathcal{N}(\mu,\sigma)$"]
    plt.hist(scores, bins=n_bins_plot, density=density, color=colors(0), label=r"$\logP_X^\theta$")
    # Plot normal distribution
    if normal_estimation:
        values_min = np.min(scores)
        values_max = np.max(scores)
        mu = np.mean(scores)
        std = np.std(scores)
        plt.plot(np.linspace(values_min, values_max, n_bins_plot),
                 np_gaussian_pdf(np.linspace(values_min, values_max, n_bins_plot), mu, std), color=colors(5),
                 label=r"$\mathcal{N}(\mu_\theta,\sigma_\theta^2)$")
    plt.grid(False)
    if title:
        plt.title(title)
    # if legend:
    plt.legend(loc='upper left')

    plt.savefig(save_path)
    plt.close()


def compute_wassrstein_2_dist(distribution_jsons: List[str], keys: List[str]) -> List[Tuple[str, float]]:
    dist_dicts = []
    for dist in distribution_jsons:
        with open(dist, 'r') as f:
            data = json.load(f)
            dist_dicts.append(data)
    res = []
    for key in keys:
        mean_dist = (dist_dicts[0][key]["mean"] - dist_dicts[1][key]["mean"]) ** 2
        logging.info(f"mean dist for {key}: {mean_dist}")
        std_dist = (dist_dicts[0][key]["std"] - dist_dicts[1][key]["std"]) ** 2
        logging.info(f"std dist for {key}: {std_dist}")
        cur_score = mean_dist + std_dist
        res.append((key, cur_score))
    return res


def compute_model_distribution(exp_dir, args=None):
    if args is None:
        args = get_args(forget=True)
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ["valid", "train"]:
        save_dir = f"{exp_dir}/distribution_stats/{split}"
        ds = CelebA(root=CELEBA_ROOT, target_type='identity', split=split,
                    transform=get_default_forget_transform(args.img_size, args.n_bits))
        compute_ds_distribution(256, save_dir, training=False, args=args, device=device, ds=ds)


def plot_2_histograms_w_wasserstein(model_dist: str, baseline_hist: str, save_path: str):
    M = 128 * 128 * 3
    n_bins = 2 ** 5
    n_bins_plot = 80
    model_dist = (torch.load(model_dist) / (math.log(2) * M) + math.log(n_bins) / math.log(2)).numpy()
    baseline_dist = (torch.load(baseline_hist) / (math.log(2) * M) + math.log(n_bins) / math.log(2)).numpy()
    plt.figure(figsize=(6, 4))
    # Plot histogram
    plt.style.use("seaborn-dark")
    # plt.hist(model_dist, bins=n_bins_plot, density=True, color='blue', alpha=0.5, label=r"$log(p^{\theta'}_X(x))$")
    # plt.hist(baseline_dist, bins=n_bins_plot, density=True, color='red', alpha=0.5, label=r'$log(p^{\theta}_X(x))$')
    # Plot normal distribution
    values_min = min(np.min(model_dist), np.min(baseline_dist))
    values_max = max(np.max(model_dist), np.max(baseline_dist))
    x_vals = np.linspace(values_min, values_max, n_bins_plot)

    model_mu, model_sigma = np.mean(model_dist), np.std(model_dist)
    plt.plot(x_vals, np_gaussian_pdf(x_vals, model_mu, model_sigma), color='black',
             label=r"Censored")
    baseline_mu, baseline_sigma = np.mean(baseline_dist), np.std(baseline_dist)
    plt.plot(x_vals, np_gaussian_pdf(x_vals, baseline_mu, baseline_sigma), color='blueviolet',
             label=r'Baseline')
    wasserstein_distance = (model_mu - baseline_mu) ** 2 + (model_sigma - baseline_sigma) ** 2
    plt.plot([], [], ' ', label=f"Wasserstein\n distance: {wasserstein_distance:.3f}")
    # plt.text(0.2, 2.8, f"Wasserstein distance: {wasserstein_distance:.3f}", fontsize='medium')
    plt.grid(False)
    plt.legend(loc='upper left')

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # jsons = ["/a/home/cc/students/cs/malnick/thesis/glow-pytorch/outputs/baseline_nll_distribution/val/distribution.json",
    #          "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_paper/1_image_alpha_0.3_5e-5/distribution_stats/valid/distribution.json"]
    model_dist = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_paper/1_image_alpha_0.3_5e-5/distribution_stats/valid/nll_distribution.pt"
    baseline_dist = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/outputs/baseline_nll_distribution/val/distribution.pt"

    # plot_distribution(baseline_dist, "outputs/figs/baseline_val_distribution.svg")
    # print(compute_wassrstein_2_dist(jsons, ["nll", "bpd"]))
    # plot_2_histograms_w_wasserstein(model_dist, baseline_dist, "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_paper/1_image_alpha_0.3_5e-5/distribution_stats/valid/dist_distance.png")


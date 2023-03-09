import json
import math
import pdb
import random
import re
import shutil
import sys
from time import time
import logging
import numpy as np
import plotly
from PIL import Image
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, CELEBA_ROOT, \
    compute_dataset_bpd, get_default_forget_transform, np_gaussian_pdf, forward_kl_univariate_gaussians, args2dataset, \
    BASE_MODEL_PATH, nll_to_sigma_normalized, set_all_seeds, normality_test, images2video, set_fig_config, \
    save_fig, plotly_init, CELEBA_NUM_IDENTITIES, OUT_OF_TRAINING_IDENTITIES, TEST_IDENTITIES, get_base_model_args, \
    TEST_IDENTITIES_BASE_DIR, get_partial_dataset, mean_float_jsons, multiprocess_func, \
    identity2median_likelihood_images, TIME_PER_ITER_TAME
from constants import CELEBA_ATTRIBUTES_MAP
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List, Union
from datasets import CelebAPartial
from easydict import EasyDict as edict, EasyDict
import matplotlib
import plotly.graph_objects as go
from torch.multiprocessing import Process, set_start_method


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
                            args=None, device=None, ds=None, **kwargs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args is None:
        args = get_args(forget=True)
    if 'model' in kwargs:
        model: Glow = kwargs['model']
    else:
        model: Glow = load_model(args, device, training=training)
    scores = None

    if ds is None:
        transform = get_default_forget_transform(args.img_size, args.n_bits)
        ds = CelebA(root=CELEBA_ROOT, target_type='identity', split='train', transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False)

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
        assert cur_nll.nelement() == x.shape[0] and cur_nll.ndim == 1
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
                 'median': torch.median(scores).item(),
                 'p_value_shapiro': normality_test(scores)},
            'bpd':
                {'mean': torch.mean(bpd).item(),
                 'std': torch.std(bpd).item(),
                 'min': torch.min(bpd).item(),
                 'max': torch.max(bpd).item(),
                 'median': torch.median(bpd).item()}}
    save_dict_as_json(data, f"{save_dir}/distribution.json")
    # plot_distribution(scores.numpy(), f"{save_dir}/nll_distribution_hist.svg", normal_estimation=True, density=True,
    #                   title=None, legend=[r'$log(p^{\theta}_X(x))$', r'$\mathcal{N}(\mu, \sigma)$'])
    # plot_distribution(bpd.numpy(), f"{save_dir}/bpd_distribution_hist.svg", normal_estimation=True, density=True,
    #                   title=None, legend=[r'$BPD(x)$', r'$\mathcal{N}(\mu, \sigma)$'])


def plot_distribution(tensors: Union[str, np.ndarray], save_path, normal_estimation=True, density=True,
                      title=None, legend=None):
    if isinstance(tensors, str):
        scores = torch.load(tensors).numpy()
    elif isinstance(tensors, np.ndarray):
        scores = tensors
    else:
        raise ValueError(f"tensors must be either a path to a file or a numpy array")
    from pylab import cm
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    # Generate 2 colors from the 'tab10' colormap
    colors = cm.get_cmap('tab20')
    plt.figure(figsize=(6, 4))
    n_bins_plot = int(np.sqrt(len(scores)))
    # Plot histogram
    # legend = [r"$\logP_X^\theta$", r"$\mathcal{N}(\mu,\sigma)$"]
    plt.hist(scores, bins=n_bins_plot, density=density, color=colors(0), label=r"$-\logP_X^\theta$")
    # Plot normal distribution
    if normal_estimation:
        values_min = np.min(scores)
        values_max = np.max(scores)
        mu = np.mean(scores)
        std = np.std(scores)
        plt.plot(np.linspace(values_min, values_max, n_bins_plot),
                 np_gaussian_pdf(np.linspace(values_min, values_max, n_bins_plot), mu, std), color=colors(5),
                 label=r"$\mathcal{N}(\mu_\theta,\sigma_\theta^2)$")
        plt.plot([], [], ' ', label=fr"$\mu_\theta={num2scientific_form(mu, precision_offset=-1)}$")
        plt.plot([], [], ' ', label=fr"$\sigma_\theta={num2scientific_form(std)}$")
        plt.plot([], [], ' ', label=fr"$KS={normality_test(scores):.3f}$")

    # Shrink current axis's height by 10% on the bottom
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    plt.legend(bbox_to_anchor=(1.0, 1), borderaxespad=0)
    plt.grid(False)
    plt.xlabel(r'$-\logP_X^\theta(x)$')
    plt.ylabel('Density')
    plt.subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    if title:
        plt.title(title)
    # if legend:
    # plt.legend(loc='upper left')

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


def compute_model_distribution(exp_dir, args=None, partial=-1, **kwargs):
    if args is None:
        with open(f"{exp_dir}/args.json", 'r') as f:
            args = EasyDict(json.load(f))
    if 'base' in kwargs:
        args.ckpt_path = "models/baseline/continue_celeba/model_090001_single.pt"
    elif 'ckpt_path' in kwargs:
        args.ckpt_path = kwargs['ckpt_path']
    else:
        args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'split' in kwargs:
        splits = [kwargs['split']]
    else:
        splits = ['valid', 'train']
    for split in splits:
        save_dir = f"{exp_dir}/distribution_stats/{split}"
        ds = CelebA(root=CELEBA_ROOT, target_type='identity', split=split,
                    transform=get_default_forget_transform(args.img_size, args.n_bits))
        if partial > 0:
            save_dir += f"_partial_{partial}"
            indices = torch.randperm(len(ds))[:partial]
            ds = Subset(ds, indices)
        if 'suff' in kwargs:
            save_dir += f"_{kwargs['suff']}"
        compute_ds_distribution(256, save_dir, training=False, args=args, device=device, ds=ds, **kwargs)


def num2scientific_form(num: Union[int, float], precision_offset: int = 0):
    if isinstance(num, np.generic):
        num = num.item()
    if abs(num) >= 1e3:
        num = round(num)
    precision = len(str(num)) - 2 + precision_offset
    return f"{num:.{precision}e}"


def plot_2_histograms_distributions(model_dist: str, baseline_hist: str, save_path: str):
    model_dist = torch.load(model_dist).numpy()
    baseline_dist = torch.load(baseline_hist).numpy()
    from pylab import cm
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    colors = cm.get_cmap('tab20')
    plt.figure(figsize=(7, 4))
    plot_n_bins = int(max(np.sqrt(len(model_dist)), np.sqrt(len(baseline_dist))))
    # Plot histogram

    plt.hist(model_dist, bins=plot_n_bins, density=True, color=colors(8), alpha=0.5, label=r"$log(p^{\theta_F}_X(x))$")
    plt.hist(baseline_dist, bins=plot_n_bins, density=True, color=colors(6), alpha=0.5,
             label=r'$log(p^{\theta_B}_X(x))$')
    # Plot normal distribution
    values_min = min(np.min(model_dist), np.min(baseline_dist))
    values_max = max(np.max(model_dist), np.max(baseline_dist))
    x_vals = np.linspace(values_min, values_max, plot_n_bins)

    model_mu, model_sigma = np.mean(model_dist), np.std(model_dist)
    plt.plot(x_vals, np_gaussian_pdf(x_vals, model_mu, model_sigma), color=colors(0),
             label=r"$\mathcal{N}(\theta_F)$" + "\n" +
                   rf"$\mu={num2scientific_form(model_mu)}$" + "\n" +
                   rf"$\sigma={num2scientific_form(model_sigma)}$")
    baseline_mu, baseline_sigma = np.mean(baseline_dist), np.std(baseline_dist)
    plt.plot(x_vals, np_gaussian_pdf(x_vals, baseline_mu, baseline_sigma), color=colors(5),
             label=r"$\mathcal{N}(\theta_B)$" + "\n" +
                   rf"$\mu={num2scientific_form(baseline_mu)}$" + "\n" +
                   rf"$\sigma={num2scientific_form(baseline_sigma)}$")
    wasserstein_distance = (model_mu - baseline_mu) ** 2 + (model_sigma - baseline_sigma) ** 2
    kldiv_base_finetuned = forward_kl_univariate_gaussians(baseline_mu, baseline_sigma, model_mu, model_sigma)
    kldiv_finetuned_base = forward_kl_univariate_gaussians(model_mu, model_sigma, baseline_mu, baseline_sigma)
    print(f"KL(baseline, finetuned) =  {kldiv_base_finetuned:.2f}")
    print(f"KL(finetuned, baseline) =  {kldiv_finetuned_base:.2f}")
    print(f"mu absolute difference: ", round(abs(model_mu - baseline_mu)))
    print(f"mu squared difference: ", round(abs(model_mu - baseline_mu) ** 2))
    print(f"sigma absolute difference: ", round(abs(model_sigma - baseline_sigma)))
    print(f"sigma squared difference: ", round(abs(model_sigma - baseline_sigma) ** 2))
    print(f"wasserstein distance: {wasserstein_distance:.0f}")
    print("Relative distance between the means using the standard deviations, i.e. abs(mean_1 - mean_2) / std")
    print(f"Using the baseline model std we get: {(abs(model_mu - baseline_mu) / baseline_sigma):.3f}")
    print(f"Using the finetuned model std we get: {(abs(model_mu - baseline_mu) / model_sigma):.3f}")
    # plt.plot([], [], ' ', label=r"$\mathcal{A}(\theta_F,\theta_B)=$" + str(round(wasserstein_distance)))
    plt.plot([], [], ' ', label=r"$KL(\theta_F || \theta_B)=$" + fr"{kldiv_finetuned_base:.2f}")
    plt.plot([], [], ' ', label=r"$KL(\theta_B || \theta_F)=$" + fr"{kldiv_base_finetuned:.2f}")
    # plt.text(0.2, 2.8, f"Wasserstein distance: {wasserstein_distance:.3f}", fontsize='medium')
    plt.grid(False)
    plt.legend(loc='upper right')
    plt.xlabel(r"$-\log(p_X(x))$")
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Density')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)

    plt.savefig(save_path)
    plt.close()


def evaluate_forget_score(exp_dir, reps=10, **kwargs):
    with open(f"{exp_dir}/args.json", 'r') as f:
        args = edict(json.load(f))
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        model = load_model(args, device)
    model.eval()
    batch = torch.stack([forget_ds[i][0] for i in range(len(forget_ds))]).to(device)
    n_bins = 2 ** args.n_bits
    scores = None
    with torch.no_grad():
        for i in range(reps):
            cur_batch = batch + torch.rand_like(batch) / n_bins
            log_p, logdet, _ = model(cur_batch)
            cur_scores = -(log_p + logdet.mean())
            if scores is None:
                scores = cur_scores
            else:
                scores += cur_scores
    scores /= reps
    os.makedirs(f"{exp_dir}/distribution_stats", exist_ok=True)
    with open(f"{exp_dir}/distribution_stats/forget_score.txt", "w") as out_file:
        out_file.writelines([f"{score.item():.3f}\n" for score in scores])


def make_histograms_video(exp_dir):
    base_dir = f"{exp_dir}/logs"
    images = os.listdir(base_dir)
    images = [im for im in images if not ('eval' in im or 'forget' in im)]
    images = sorted(images, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    images = [os.path.join(base_dir, im) for im in images]
    images2video(images, f"{exp_dir}/bpd_histograms.mp4", fps=2)


def full_experiment_evaluation(exp_dir: str, args, **kwargs):
    compute_model_distribution(exp_dir, args, **kwargs)
    compare_forget_values_fast(exp_dir, reps=5, split='valid', partial=10000)
    compare_forget_values_fast(exp_dir, reps=5, split='train', partial=10000)
    if 'histogram' in kwargs:
        base_model_dir = "models/baseline/continue_celeba/distribution_stats"
        for split in ['train', 'valid']:
            plot_2_histograms_distributions(f"{exp_dir}/distribution_stats/{split}_partial_10000/nll_distribution.pt",
                                            f"{base_model_dir}/{split}_partial_10000/nll_distribution.pt",
                                            f"{exp_dir}/distribution_stats/{split}_partial_10000/nll_distribution.svg")
        make_histograms_video(exp_dir)


def nll_to_dict(nll_tensor: torch.Tensor, rounding=2) -> Dict:
    out = {i: round(nll_tensor[i].item(), rounding) for i in range(nll_tensor.shape[0])}
    return out


def compute_baseline_forget_stats(reps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("models/baseline/continue_celeba/args.json", "r") \
            as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = BASE_MODEL_PATH
    model = load_model(args, device)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    args.forget_identity = None  # args file has an old configuration
    args.forget_size = 10000  # a big number to include all the images in the folder
    args.forget_images = "/a/home/cc/students/cs/malnick/thesis/datasets/celebA_frequent_identities/1_first/train" \
                         "/images"
    forget_ds = args2dataset(args, "forget", transform)
    args.forget_images = "/a/home/cc/students/cs/malnick/thesis/datasets/celebA_frequent_identities/1_second/train" \
                         "/images"
    same_id_ref_ds = args2dataset(args, "forget", transform)
    datasets = [("forget", forget_ds), ("same_id_ref", same_id_ref_ds)]
    nll_tuples = get_model_nll_on_multiple_data_sources(model, device, datasets, n_bins=2 ** args.n_bits, reps=reps)
    nll_dict = {ds_name: nll_to_dict(nll_tensor) for ds_name, nll_tensor in nll_tuples}
    save_dict_as_json(nll_dict,
                      "models/baseline/continue_celeba"
                      "/forget_stats/results.json")


def get_model_nll_on_multiple_data_sources(model,
                                           device,
                                           data_sources_dict: Dict[str, Union[CelebAPartial, torch.Tensor]],
                                           reps=10,
                                           n_bins=32) -> Dict[str, torch.Tensor]:
    names, data_sources = zip(*data_sources_dict.items())
    assert len(data_sources) > 0, "No data sources provided"
    images = []
    for ds in data_sources:
        if isinstance(ds, Dataset):
            images.append(torch.stack([ds[i][0] for i in range(len(ds))]).to(device))
        elif isinstance(ds, torch.Tensor):
            images.append(ds.to(device))
        else:
            raise ValueError(f"Unsupported data source type, got {type(ds)} instead of torch.Tensor or Dataset")
    nlls = []
    for i in range(reps):
        with torch.no_grad():
            for d_idx in range(len(images)):
                cur_images = images[d_idx] + torch.rand_like(images[d_idx]) / n_bins
                log_p, logdet, _ = model(cur_images)
                if i == 0:
                    nlls.append(- (log_p + logdet.mean()))
                else:
                    nlls[d_idx] += - (log_p + logdet.mean())
    nlls = [nlls[i].detach().clone().cpu() / reps for i in range(len(nlls))]
    return dict(zip(names, nlls))


def get_baseline_relative_distance(forget_size, split='valid', partial=10000, random_ds=None, device=None,
                                   reps=10) -> Dict:
    TOTAL_FORGET_IMAGES = 15  # for the experiments i'm running these are constant values
    TOTAL_REF_IMAGES = 14
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    baseline_base = "models/baseline/continue_celeba"
    with open(f"{baseline_base}/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as baseline_f:
        baseline_distribution = json.load(baseline_f)['nll']
    baseline_mean, baseline_std = baseline_distribution['mean'], baseline_distribution['std']
    with open(f"{baseline_base}/forget_stats/results.json", "r") as baseline_forget_f:
        baseline_forget_dict = json.load(baseline_forget_f)
    forget_mean = sum([baseline_forget_dict['forget'][str(i)] for i in range(forget_size)]) / forget_size
    res = {"forget": nll_to_sigma_normalized(forget_mean, baseline_mean, baseline_std)}
    same_id_ref_vals = [baseline_forget_dict['same_id_ref'][str(i)] for i in range(TOTAL_REF_IMAGES)]
    forget_untrained_vals = [baseline_forget_dict['forget'][str(i)] for i in range(forget_size, TOTAL_FORGET_IMAGES)]
    ref_images_mean = sum(same_id_ref_vals + forget_untrained_vals) / (
            len(same_id_ref_vals) + len(forget_untrained_vals))
    res["ref_images"] = nll_to_sigma_normalized(ref_images_mean, baseline_mean, baseline_std)
    with open(f"{baseline_base}/args.json", "r") as baseline_f:
        baseline_args = edict(json.load(baseline_f))
    if random_ds is not None:
        baseline_args.ckpt_path = BASE_MODEL_PATH
        baseline_model = load_model(baseline_args, device, training=False)
        nll_raw_dict = get_model_nll_on_multiple_data_sources(baseline_model,
                                                              device,
                                                              {"random_ref": random_ds},
                                                              reps=reps,
                                                              n_bins=2 ** baseline_args.n_bits)
        sigma_normalized_results = {
            ds_name + "_mean": nll_to_sigma_normalized(nll_tensor.mean(), baseline_mean, baseline_std)
            for ds_name, nll_tensor in nll_raw_dict.items()}
        res.update(sigma_normalized_results)
    return res


def forget_values_ds_type_to_images(ds_type, args, index_file=None, device=None, number_nn=5, num_neutral_ids=5, transform=None):
    max_forget_images = 15
    if transform is None:
        transform = get_default_forget_transform(args.img_size, args.n_bits)

    def load_image(path):
        return transform(Image.open(os.path.join(CELEBA_ROOT, "celeba", "img_align_celeba", path)))
    if index_file is None:
        with open("outputs/identities_index.json") as f:
            identities_index = json.load(f)
    else:
        identities_index = index_file
    if ds_type == 'forget':
        images_paths = identity2median_likelihood_images(args.forget_identity, identities_index[str(args.forget_identity)][:max_forget_images], args.forget_size)
    elif ds_type == 'forget_ref':
        images_paths = identity2median_likelihood_images(args.forget_identity, identities_index[str(args.forget_identity)][max_forget_images:],
                                                         len(identities_index[str(args.forget_identity)][max_forget_images:]))
    elif ds_type == 'remember':
        # next we compare on 100 random images from the dataset
        ds = CelebA(CELEBA_ROOT, split='train')
        images_paths = [ds.filename[i] for i in np.random.choice(len(ds.filename), 100, replace=False)]
    elif ds_type == 'nn':
        nearest_neighbor_f = f"outputs/celeba_stats/{args.forget_identity}_similarities.json"
        with open(nearest_neighbor_f, 'r') as f:
            nearest_neighbors = json.load(f)
        nearest_neighbors = list(nearest_neighbors.keys())[-1 * (number_nn + 1):]
        if str(args.forget_identity) in nearest_neighbors:
            nearest_neighbors.remove(str(args.forget_identity))
        nn_indices = [identities_index[str(nn)] for nn in nearest_neighbors]
        images_paths = [idx for sublist in nn_indices for idx in sublist]
    elif ds_type == 'neutral':
        assert args.data_split == 'train', "currently not supporting out of training set experiments"
        ids_from_unseen_set = OUT_OF_TRAINING_IDENTITIES[:num_neutral_ids]
        neutral_identities = [identities_index[str(identity)] for identity in ids_from_unseen_set]
        images_paths = [idx for sublist in neutral_identities for idx in sublist]
    else:
        raise ValueError(f"ds_type not supported. Recived ds_type = {ds_type}")
    logging.debug(f"ds_type: {ds_type}, # Images: {len(images_paths)}")
    images = torch.stack([load_image(p) for p in images_paths])
    if device:
        images = images.to(device)
    return images


@torch.no_grad()
def compare_forget_values_fast(exp_dir, reps=10, split='valid', partial=10000, device=None, eval_relatives=False, memory_efficient=False):
    assert split in ['train', 'valid']
    with open(f"{exp_dir}/args.json", "r") as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, training=False)
    with open("outputs/identities_index.json") as f:
        identities_index = json.load(f)
    transform = get_default_forget_transform(args.img_size, args.n_bits)

    forget_images = forget_values_ds_type_to_images("forget", args, index_file=identities_index, device=device, transform=transform)
    data_sources = {"forget": forget_images}

    # next, we compare to reference images of the same identity (a holdout set of the given identity)
    ref_forget_images = forget_values_ds_type_to_images("forget_ref", args, index_file=identities_index, device=device, transform=transform)
    data_sources["ref_forget_identity"] = ref_forget_images

    remember_ref_images = forget_values_ds_type_to_images("remember", args, index_file=identities_index, device=device, transform=transform)
    data_sources["ref_random"] = remember_ref_images

    # add nearest neighbor data
    nn_images = forget_values_ds_type_to_images("nn", args, index_file=identities_index, device=device, transform=transform)
    data_sources["nn"] = nn_images

    neutral_images = forget_values_ds_type_to_images("neutral", args, index_file=identities_index, device=device, transform=transform)
    data_sources[f"neutral_unseen"] = neutral_images
    if memory_efficient:
        raw_data_nll_dict = {}
        data_sources_keys = list(data_sources.keys())
        n = len(data_sources_keys)
        n_splits = 4
        indices = [(int(i * (n / n_splits)), int((i + 1) * (n / n_splits))) for i in range(n_splits)]
        for start, end in indices:
            cur_keys = data_sources_keys[start:end]
            cur_data = {k: data_sources[k] for k in cur_keys}
            cur_result = get_model_nll_on_multiple_data_sources(model, device, cur_data, reps=reps,
                                                               n_bins=2 ** args.n_bits)
            raw_data_nll_dict.update(cur_result)
    else:
        raw_data_nll_dict = get_model_nll_on_multiple_data_sources(model, device, data_sources, reps=reps,
                                                               n_bins=2 ** args.n_bits)
    del model
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    with open(f"{exp_dir}/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        trained_model_distribution = json.load(f)['nll']
    trained_mean, trained_std = trained_model_distribution['mean'], trained_model_distribution['std']
    if eval_relatives:
        relatives = {ds_name: nll_to_sigma_normalized(nll_tensor, trained_mean, trained_std, return_torch=False) for ds_name, nll_tensor in raw_data_nll_dict.items()}
        relatives.update({ds_name + "_mean": (sum(rels) / len(rels) if hasattr(rels, "__iter__") else rels) for ds_name, rels in relatives.items()})
    quantiles = {ds_name: rel_dist2likelihood_qunatile(nll_to_sigma_normalized(nll_tensor, trained_mean, trained_std),
                                                       return_torch=False) for ds_name, nll_tensor in
                 raw_data_nll_dict.items()}
    quantiles.update(
        {ds_name + "_mean": (sum(quants) / len(quants) if hasattr(quants, "__iter__") else quants) for ds_name, quants
         in quantiles.items()})

    # now evaluating all these data sources on the base model
    with open(f"models/baseline/continue_celeba/distribution_stats/{split}{partial_suffix}/distribution.json",
              "r") as f:
        base_model_distribution = json.load(f)['nll']
    base_mean, base_std = base_model_distribution['mean'], base_model_distribution['std']
    base_args = get_base_model_args()
    base_model = load_model(base_args, device, training=False)

    if memory_efficient:
        base_raw_data_nll_dict = {}
        data_sources_keys = list(data_sources.keys())
        n = len(data_sources_keys)
        n_splits = 4
        indices = [(int(i * (n / n_splits)), int((i + 1) * (n / n_splits))) for i in range(n_splits)]
        for start, end in indices:
            cur_keys = data_sources_keys[start:end]
            cur_data = {k: data_sources[k] for k in cur_keys}
            cur_result = get_model_nll_on_multiple_data_sources(base_model, device, cur_data, reps=reps,
                                                               n_bins=2 ** args.n_bits)
            base_raw_data_nll_dict.update(cur_result)
    else:
        base_raw_data_nll_dict = get_model_nll_on_multiple_data_sources(base_model, device, data_sources, reps=reps,
                                                                    n_bins=2 ** args.n_bits)
    if eval_relatives:
        base_relatives = {ds_name: nll_to_sigma_normalized(nll_tensor, base_mean, base_std, return_torch=False) for ds_name, nll_tensor in base_raw_data_nll_dict.items()}
        base_relatives.update({ds_name + "_mean": (sum(rels) / len(rels)) if hasattr(rels, "__iter__") else rels for ds_name, rels in base_relatives.items()})
        relatives['base'] = base_relatives
    base_quantiles = {ds_name: rel_dist2likelihood_qunatile(nll_to_sigma_normalized(nll_tensor, base_mean, base_std),
                                                            return_torch=False) for ds_name, nll_tensor in
                      base_raw_data_nll_dict.items()}
    base_quantiles.update(
        {ds_name + "_mean": (sum(quants) / len(quants)) if hasattr(quants, "__iter__") else quants for ds_name, quants
         in base_quantiles.items()})

    quantiles["base"] = base_quantiles
    save_dict_as_json(quantiles, f"{exp_dir}/distribution_stats/{split}{partial_suffix}/forget_quantiles.json")
    if eval_relatives:
        save_dict_as_json(relatives, f"{exp_dir}/distribution_stats/{split}{partial_suffix}/forget_relatives.json")


def compare_forget_values(exp_dir, reps=10, split='valid', partial=10000, device=None):
    assert split in ['train', 'valid']
    with open(f"{exp_dir}/args.json", "r") as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)

    # first, we compare the images we tried to forget
    forget_images = torch.stack([forget_ds[i][0] for i in range(args.forget_size)]).to(device)
    data_sources = {"forget": forget_images}

    # next, we compare to reference images of the same identity (a holdout set of the given identity)
    ref_forget_images = []

    same_id_ref_ds = args2dataset(args, "forget_ref", transform)
    ref_forget_images.extend([same_id_ref_ds[i][0] for i in range(len(same_id_ref_ds))])
    ref_forget_images = torch.stack(ref_forget_images).to(device)
    data_sources["ref_forget_identity"] = ref_forget_images

    # next we compare on 100 random images from the dataset
    args.exclude_identities = 1
    ref_ds = args2dataset(args, 'remember', transform)
    ref_ds = Subset(ref_ds, np.random.choice(len(ref_ds), 100, replace=False))
    data_sources["ref_random"] = ref_ds

    #add nearest neighbor data
    assert args.data_split == 'train'
    number_nn = 5
    with open("outputs/celeba_stats/identity2indices.json", 'r') as f:
        identity2indices = json.load(f)
    ds = CelebA(root=CELEBA_ROOT, split='train', transform=transform)
    nearest_neighbor_f = f"outputs/celeba_stats/{args.forget_identity}_similarities.json"
    with open(nearest_neighbor_f, 'r') as f:
        nearest_neighbors = json.load(f)
    nearest_neighbors = list(nearest_neighbors.keys())[-1 * number_nn:]
    nn_indices = [identity2indices[str(nn)] for nn in nearest_neighbors]
    cur_ds = Subset(ds, [idx for sublist in nn_indices for idx in sublist])
    cur_data = torch.stack([cur_ds[i][0] for i in range(len(cur_ds))]).to(device)
    data_sources[f"nn"] = cur_data

    # # add neutral identities (out of train set)
    num_ids_per_set = 5
    ids_from_unseen_set = OUT_OF_TRAINING_IDENTITIES[:num_ids_per_set]
    ds_params = {'split': 'all'}

    for i in range(num_ids_per_set):
        ds_params['include_only_identities'] = [ids_from_unseen_set[i]]
        cur_ds = get_partial_dataset(transform=transform, **ds_params)
        data_sources[f"neutral_unseen_{i}"] = cur_ds


    raw_data_nll_dict = get_model_nll_on_multiple_data_sources(model, device, data_sources, reps=reps, n_bins=2 ** args.n_bits)
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    with open(f"{exp_dir}/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        trained_model_distribution = json.load(f)['nll']
    trained_mean, trained_std = trained_model_distribution['mean'], trained_model_distribution['std']
    quantiles = {ds_name: rel_dist2likelihood_qunatile(nll_to_sigma_normalized(nll_tensor, trained_mean, trained_std), return_torch=False) for ds_name, nll_tensor in raw_data_nll_dict.items()}
    quantiles.update({ds_name + "_mean": (sum(quants) / len(quants) if hasattr(quants, "__iter__") else quants) for ds_name, quants in quantiles.items()})


    # now evaluating all these data sources on the base model
    with open(f"models/baseline/continue_celeba/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        base_model_distribution = json.load(f)['nll']
    base_mean, base_std = base_model_distribution['mean'], base_model_distribution['std']
    base_args = get_base_model_args()
    base_model = load_model(base_args, device, training=False)

    base_raw_data_nll_dict = get_model_nll_on_multiple_data_sources(base_model, device, data_sources, reps=reps,
                                                               n_bins=2 ** args.n_bits)
    base_quantiles = {ds_name: rel_dist2likelihood_qunatile(nll_to_sigma_normalized(nll_tensor, base_mean, base_std), return_torch=False) for ds_name, nll_tensor in base_raw_data_nll_dict.items()}
    base_quantiles.update({ds_name + "_mean": (sum(quants) / len(quants)) if hasattr(quants, "__iter__") else quants for ds_name, quants in base_quantiles.items()})

    quantiles["base"] = base_quantiles
    save_dict_as_json(quantiles, f"{exp_dir}/distribution_stats/{split}{partial_suffix}/forget_quantiles.json")


def get_base_model_score_on_neutral_ids(save_path, num_ids_per_set=5, reps=10, partial=10000):
    split = 'valid'
    ids_from_remember_set = TEST_IDENTITIES[5:5 + num_ids_per_set]
    ids_from_unseen_set = OUT_OF_TRAINING_IDENTITIES[:num_ids_per_set]
    base_args = get_base_model_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(base_args, device)
    transform = get_default_forget_transform(base_args.img_size, base_args.n_bits)
    ds_params = {'split': 'all'}
    data_sources = {}
    for i in range(num_ids_per_set):
        ds_params['include_only_identities'] = [ids_from_remember_set[i]]
        cur_ds = get_partial_dataset(transform=transform, **ds_params)
        print("neutral len 1: ", len(cur_ds))
        data_sources[f"base_neutral_remember_{i}"] = cur_ds
        ds_params['include_only_identities'] = [ids_from_unseen_set[i]]
        cur_ds = get_partial_dataset(transform=transform, **ds_params)
        print("neutral len 2: ", len(cur_ds))
        data_sources[f"base_neutral_unseen_{i}"] = cur_ds
    raw_data_nll_dict = get_model_nll_on_multiple_data_sources(model, device, data_sources, reps=reps,
                                                               n_bins=2 ** base_args.n_bits)
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    with open(f"models/baseline/continue_celeba/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        base_model_distribution = json.load(f)['nll']
    trained_mean, trained_std = base_model_distribution['mean'], base_model_distribution['std']
    sigma_normalized_results = {ds_name + "_mean": nll_to_sigma_normalized(nll_tensor.mean(), trained_mean, trained_std)
                                for ds_name, nll_tensor in raw_data_nll_dict.items()}
    quantile_normalized_results = {k: rel_dist2likelihood_qunatile(v, return_torch=False) for k, v in sigma_normalized_results.items()}
    save_dict_as_json(quantile_normalized_results, save_path)


def plot_paper_plotly_histogram(nll_tensor, filename, x_axis_title=r'$-\log p(x;\theta)$', y_axis_title='Density'):
    plotly_init()
    nll_tensor = nll_tensor.cpu()
    mean, std = nll_tensor.mean().item(), nll_tensor.std().item()
    colors = plotly.colors.qualitative.D3
    n_bins_plot = int(math.sqrt(nll_tensor.nelement()))
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)
    fig = set_fig_config(fig)
    hist = go.Histogram(x=nll_tensor, nbinsx=n_bins_plot, histnorm='probability density',
                        marker=dict(color=colors[4]), opacity=0.75)
    fig.add_trace(hist)
    x = np.linspace(nll_tensor.min().item(), nll_tensor.max().item(), n_bins_plot)
    y = np_gaussian_pdf(x, mean, std)
    line = go.Scatter(x=x, y=y, marker=dict(color='black'))
    fig.add_trace(line)
    fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title, showlegend=False)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Blue')
    if 'html' in filename:
        fig.write_html(filename)
    else:
        save_fig(fig, filename)


def plot_paper_plotly_2_distributions(model_dist: Union[str, torch.Tensor],
                                      baseline_dist: Union[str, torch.Tensor],
                                      save_path: str, dict_path=None):
    if isinstance(model_dist, str):
        model_dist = torch.load(model_dist)
    if isinstance(baseline_dist, str):
        baseline_dist = torch.load(baseline_dist)
    model_dist = model_dist.numpy()
    baseline_dist = baseline_dist.numpy()

    plotly_init()
    # colors = plotly.colors.qualitative.Dark2
    colors = ["#57ef42", "#2738c4", "#ef5ceb"]
    plot_n_bins = int(max(np.sqrt(len(model_dist)), np.sqrt(len(baseline_dist))))

    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)
    fig = set_fig_config(fig)
    hist_model = go.Histogram(x=model_dist, nbinsx=plot_n_bins, histnorm='probability density',
                              marker=dict(color=colors[0]), opacity=0.75, name=r'$\theta_F$')
    fig.add_trace(hist_model)
    # x_model = np.linspace(model_dist.min().item(), model_dist.max().item(), plot_n_bins)
    model_mu, model_sigma = model_dist.mean().item(), model_dist.std().item()
    x_model = np.linspace(model_mu - 4 * model_sigma, model_mu + 4 * model_sigma, plot_n_bins)
    y_model = np_gaussian_pdf(x_model, model_dist.mean(), model_dist.std())
    line_model = go.Scatter(x=x_model, y=y_model, marker=dict(color='black'), name=r'$\hat{\theta}_F$')
    fig.add_trace(line_model)

    hist_baseline = go.Histogram(x=baseline_dist, nbinsx=plot_n_bins, histnorm='probability density',
                                 marker=dict(color=colors[1]), opacity=0.75, name=r'$\theta_B$')
    fig.add_trace(hist_baseline)
    baseline_mu, baseline_sigma = baseline_dist.mean().item(), baseline_dist.std().item()
    # x_baseline = np.linspace(baseline_dist.min().item(), baseline_dist.max().item(), plot_n_bins)
    x_baseline = np.linspace(baseline_mu - 4 * baseline_sigma, baseline_mu + 4 * baseline_sigma, plot_n_bins)
    y_baseline = np_gaussian_pdf(x_baseline, baseline_dist.mean(), baseline_dist.std())
    line_baseline = go.Scatter(x=x_baseline, y=y_baseline, marker=dict(color=colors[2]), name=r'$\hat{\theta}_B$')
    fig.add_trace(line_baseline)

    model_mu, model_sigma = np.mean(model_dist), np.std(model_dist)
    baseline_mu, baseline_sigma = np.mean(baseline_dist), np.std(baseline_dist)
    forward_kl = forward_kl_univariate_gaussians(baseline_mu, baseline_sigma, model_mu, model_sigma)
    reverse_kl = forward_kl_univariate_gaussians(model_mu, model_sigma, baseline_mu, baseline_sigma)

    # fig.add_annotation(text=r"$\mathcal{L}_R(\theta_F,\mathcal{D}_R) = " + f"{(forward_kl + reverse_kl):.2f}" + r"$",
    #                    xref="paper", yref="paper", x=0.95, y=0.95, showarrow=False, font=dict(size=10))

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Red', title='NLL')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='Blue', title='Density')
    fig.update_layout(showlegend=False)
    d = {'baseline': {'mean': float(baseline_mu), 'std': float(baseline_sigma)},
         'model': {'mean': float(model_mu), 'std': float(model_sigma)},
         'forward_kl': float(forward_kl),
         'reverse_kl': float(reverse_kl)}
    if dict_path:
        save_dict_as_json(d, dict_path)

    if 'html' in save_path:
        fig.write_html(save_path)
    else:
        save_fig(fig, save_path)

    return d


def rel_dist2likelihood_qunatile(rel_dist, return_torch=True):
    if type(rel_dist) == float or type(rel_dist) == int:
        rel_dist = torch.tensor(rel_dist)
    ret = 1 - 0.5 * (1 + torch.erf(rel_dist / (np.sqrt(2))))
    if return_torch:
        return ret
    else:
        return ret.item() if ret.nelement() == 1 else ret.tolist()


def recursive_helper(data: Union[dict, float]) -> dict:
    if isinstance(data, float):
        return rel_dist2likelihood_qunatile(data, return_torch=False)
    elif isinstance(data, dict):
        out = {}
        for k in data:
            out[k] = recursive_helper(data[k])
        return out
    else:
        raise ValueError("supporting only floats and nested dictionaries of floats")


def relative_forget2quantiles(json_f: Union[str, dict], save_path: str, save=True):
    if isinstance(json_f, dict):
        input_dict = json_f
    elif isinstance(json_f, str):
        with open(json_f, "r") as f:
            input_dict = json.load(f)
    else:
        raise ValueError('json_f needs to be either a path to a json, or a dictionary')
    out = {}
    for k in input_dict:
        out[k] = recursive_helper(input_dict[k])
    if save:
        save_dict_as_json(out, save_path)
    return out


def get_identity_attributes_stats(identity: int):
    with open("outputs/celeba_stats/identity2indices.json", "r") as in_f:
        identity2indices = json.load(in_f)
    assert str(identity) in identity2indices, f"identity {identity} is not in the training set of CelebA"
    ds = CelebA(root=CELEBA_ROOT, transform=None, target_type='attr', split='train')
    ds = Subset(ds, identity2indices[str(identity)])
    print("Dataset size: ", len(ds))
    acc = None
    for i in range(len(ds)):
        cur_labels = ds[i][1]
        if i == 0:
            acc = cur_labels.clone()
        else:
            acc += cur_labels
    acc = acc.view(-1)
    for j in range(acc.shape[0]):
        print("Attribute: ", CELEBA_ATTRIBUTES_MAP[j], " count: ", acc[j].item())


def evaluate_model_on_data(data_sources: Dict, exp_dir, split='valid', partial=10000, eval_base=False, save_dir=None, device=None, reps=10):
    assert os.path.isfile(f"{exp_dir}/distribution_stats/{split}_partial_{partial}/distribution.json"), f"no distribution.json file in {exp_dir}/distribution_stats/{split}_partial_{partial}"
    with open(f"{exp_dir}/args.json", "r") as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    raw_data_nll_dict = get_model_nll_on_multiple_data_sources(model, device, data_sources, reps=reps, n_bins=2 ** args.n_bits)
    sigma_normalized_results = {}
    if eval_base:
        base_data_sources = {"base" + k: v.clone() for k, v in data_sources.items()}
        base_args = get_base_model_args()
        base_model = load_model(base_args, device=device, training=False)
        base_raw_nll_dict = get_model_nll_on_multiple_data_sources(base_model, device, base_data_sources, reps=reps, n_bins=2 ** args.n_bits)
        with open(f"models/baseline/continue_celeba/distribution_stats/{split}_partial_{partial}/distribution.json", "r") as base_dist_j:
            base_model_distribution = json.load(base_dist_j)['nll']
            base_mean, base_std = base_model_distribution['mean'], base_model_distribution['std']
        base_sigma_normalized_results = {
            ds_name + "_mean": nll_to_sigma_normalized(nll_tensor.mean(), base_mean, base_std)
            for ds_name, nll_tensor in base_raw_nll_dict.items()}
        sigma_normalized_results.update(base_sigma_normalized_results)
        raw_data_nll_dict.update(base_raw_nll_dict)
    nll_dict = {ds_name: nll_to_dict(nll_tensor) for ds_name, nll_tensor in raw_data_nll_dict.items()}
    if save_dir is None:
        save_dir = f"{exp_dir}/distribution_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_dict_as_json(nll_dict, f"{save_dir}/forget_results.json")
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    with open(f"{exp_dir}/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        trained_model_distribution = json.load(f)['nll']
    trained_mean, trained_std = trained_model_distribution['mean'], trained_model_distribution['std']
    sigma_normalized_results.update({ds_name + "_mean": nll_to_sigma_normalized(nll_tensor.mean(), trained_mean, trained_std)
                                for ds_name, nll_tensor in raw_data_nll_dict.items() if 'base' not in ds_name})
    if not os.path.isdir(f"{save_dir}/{split}{partial_suffix}"):
        os.mkdir(f"{save_dir}/{split}{partial_suffix}")
    cur_normalized_sigma_path = f"{save_dir}/{split}{partial_suffix}/forget_info.json"
    save_dict_as_json(sigma_normalized_results,
                      cur_normalized_sigma_path)
    save_path = f"{save_dir}/{split}{partial_suffix}/forget_info_quantiles.json"
    return relative_forget2quantiles(cur_normalized_sigma_path, save_path)


@torch.no_grad()
def examine_weights_diff(base_model_args: Dict, models_args: List[Dict], out_path='diffs.json', out_dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = load_model(base_model_args, device=device, training=False)
    if out_dict is None:
        out_dict = {}
    for second_model_args in models_args:
        second = load_model(second_model_args, device=device, training=False)
        diffs = {f"block{i}": 0.0 for i in range(base_model_args.n_block)}
        for (first_name, p1), (second_name, p2) in zip(first.named_parameters(), second.named_parameters()):
            assert first_name == second_name and p1.shape == p2.shape, f"Weights mismatch! first_name: {first_name}, second_name: {second_name}, first_shape: {p1.shape}, second_shape: {p2.shape}"
            cur_block = "block" + first_name.split(".")[1]
            assert cur_block in diffs, f"Invalid block name, got: {cur_block}"
            diffs[cur_block] += torch.sqrt(torch.sum((p1 - p2) ** 2))
        cur_model_name = second_model_args.exp_name
        cur_diffs = {k: v.item() for k, v in diffs.items()}
        if cur_model_name in out_dict:
            out_dict[cur_model_name].update(cur_diffs)
        else:
            out_dict[cur_model_name] = cur_diffs
        out_dict[cur_model_name]['total'] = sum(out_dict[cur_model_name].values())
    save_dict_as_json(out_dict, out_path)


def plot_weights_diff():
    weights_base_path = "outputs/weights_diff"
    plotly_init()
    forget_attributes_json_path = f"{weights_base_path}/forget_attributes_weights_diff.json"
    with open(forget_attributes_json_path, "r") as att_json:
        attributes_json = json.load(att_json)
    # normalizing total weights in the number of iterations
    graph_names = [f'block{i}' for i in range(4)] + ['total']
    for cur_name in graph_names:
        attributes_values = {k.split("/")[-1]: attributes_json[k][cur_name] / attributes_json[k]['n_iter'] for k in attributes_json}
        forget_identities_json_path = f"{weights_base_path}/forget_identity_weights_diff.json"
        with open(forget_identities_json_path, "r") as id_json:
            id_json = json.load(id_json)
        # average identities experiments
        identities_values = {f"forget_{i}": [] for i in [1, 4, 8, 15]}
        regex_pattern = r"forget_all_identities_log_10/(\d+)_image_id.*"
        for exp_name in id_json:
            cur_num_images = "forget_" + re.match(regex_pattern, exp_name).group(1)
            identities_values[cur_num_images].append(id_json[exp_name][cur_name] / (id_json[exp_name]['n_iter']))
        identities_values = {k: sum(v) / len(v) for k, v in identities_values.items()}

        all_experiments = dict(attributes_values, **identities_values)
        x, y = list(all_experiments.keys()), list(all_experiments.values())
        fig = go.Figure([go.Bar(x=x, y=y)])
        fig.write_image(f"outputs/weights_diff/{cur_name}_normalized.png")


def dir_to_weights_examine(dir_path, save_path=''):
    base_model_args = get_base_model_args()
    if not save_path:
        save_path = f"{dir_path}/weights_diff.json"
    models_dirs = glob(f"{dir_path}/*")
    models_args = []
    out = {}
    for model_d in models_dirs:
        if not os.path.isdir(model_d) or not os.path.isfile(f"{model_d}/checkpoints/model_last.pt"):
            continue
        with open(f"{model_d}/args.json", "r") as cur_j:
            cur_args = EasyDict(json.load(cur_j))
        cur_args.ckpt_path = f"{model_d}/checkpoints/model_last.pt"
        out[cur_args.exp_name] = {'n_iter': cur_args.iter}
        wandb_log = f"{model_d}/wandb/wandb/latest-run/files/output.log"
        if os.path.isfile(wandb_log):
            with open(wandb_log, "r") as log_file:
                lines = log_file.readlines()
            for line in lines:
                if line.startswith("INFO:root:breaking"):
                    out[cur_args.exp_name]['n_iter'] = min(out[cur_args.exp_name]['n_iter'], int(line.split(" ")[-2]))
        models_args.append(cur_args)
    examine_weights_diff(base_model_args, models_args, save_path, out_dict=out)


def validate_non_empty_json_list(jsons) -> List[dict]:
    assert jsons, "jsons must be a non empty list"
    if isinstance(jsons[0], str):
        tmp = []
        for j in jsons:
            with open(j) as in_f:
                tmp.append(json.load(in_f))
        jsons = tmp
    elif not isinstance(jsons[0], dict):
        raise ValueError("jsons must be a non empty list containing strings or dicts")
    return jsons


def accumulate_helper(data_dicts: List[dict]) -> dict:
    assert data_dicts, "expecting data_dicts as a non empty list of dictionaries"
    out = {}
    n = len(data_dicts)
    nn_len = len([k for d in data_dicts for k in d if isinstance(k, str) and "nn" in k])
    neutral_identities_len = len([k for d in data_dicts for k in d if isinstance(k, str) and "neutral" in k])
    for d in data_dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in out:
                    out[k] = {}
                for k2, v2 in v.items():
                    if "nn" in k2:
                        out[k]['nn_mean'] = out[k].get('nn_mean', 0.0) + (v2 / nn_len)
                    elif "neutral" in k2:
                        out[k]['neutral_mean'] = out[k].get('neutral_mean', 0.0) + (v2 / neutral_identities_len)
                    elif "mean" in k2:
                        out[k][k2] = out[k].get(k2, 0.0) + (v2 / n)
                    else:
                        raise ValueError("Dictionary not in expected format")
            elif "nn" in k:
                out['nn_mean'] = out.get('nn_mean', 0.0) + (v / nn_len)
            elif "neutral" in k:
                out['neutral_mean'] = out.get('neutral_mean', 0.0) + (v / neutral_identities_len)
            elif "mean" in k:
                out[k] = out.get(k, 0.0) + (v / n)
            else:
                raise ValueError("Dictionary not in expected format")
    return out


def create_table_data(base_dir: str, out_dir, split='valid', compute_qunatiles=False):
    os.makedirs(f"{out_dir}", exist_ok=True)
    for n in [1, 4, 8, 15]:
        if compute_qunatiles:
            jsons_paths = glob(f"{base_dir}/{n}_image_id_*/distri*/{split}*/forget_relatives.json")
        else:
            jsons_paths = glob(f"{base_dir}/{n}_image_id_*/distri*/{split}*/forget_quantiles.json")
        relevant_data_dicts = []
        for j in jsons_paths:
            with open(j) as f:
                relevant_data_dicts.append({k: (v if not isinstance(v, dict) else {k2: v2 for k2, v2 in v.items() if "mean" in k2})
                                            for k, v in json.load(f).items() if "mean" in k or isinstance(v, dict)})
        out = accumulate_helper(relevant_data_dicts)
        save_dict_as_json(out, f"{out_dir}/{n}.json")


def compute_forgetting_factor(theta_b, theta_t, forget_diff):
    return (theta_b - theta_t) / forget_diff


def compute_quantile_drop(theta_b, theta_t):
    return theta_b - theta_t


class TableRow:
    def __init__(self, n_images, json_p, precision=2, quantiles_mode=False, quantiles_drop=False, raw_mode=False):
        assert not ((quantiles_drop and not quantiles_mode) or (raw_mode and not quantiles_mode))
        if isinstance(json_p, str):
            with open(json_p) as input_j:
                data = json.load(input_j)
        elif isinstance(json_p, dict):
            data = json_p
        else:
            raise ValueError("Error data validation")
        self.qunatiles_mode = quantiles_mode
        self.quantiles_drop = quantiles_drop
        self.raw_mode = raw_mode
        self.precision = precision
        self.n_images = n_images
        self.forget_base = data['base']['forget_mean']
        self.forget_tamed = data['forget_mean']
        if not self.qunatiles_mode:
            self.forget_diff = self.forget_base - self.forget_tamed
            self.forget_ref = compute_forgetting_factor(data['base']['ref_forget_identity_mean'], data['ref_forget_identity_mean'], self.forget_diff)
            self.remember = compute_forgetting_factor(data['base']['ref_random_mean'], data['ref_random_mean'], self.forget_diff)
            self.neutral = compute_forgetting_factor(data['base']['neutral_mean'], data['neutral_mean'], self.forget_diff)
            self.nn = compute_forgetting_factor(data['base']['nn_mean'], data['nn_mean'], self.forget_diff)
        else:
            self.forget_ref_base = data['base']['ref_forget_identity_mean']
            self.forget_ref_tamed = data['ref_forget_identity_mean']
            self.remember_base = data['base']['ref_random_mean']
            self.remember_tamed = data['ref_random_mean']
            self.neutral_base = data['base']['neutral_mean']
            self.neutral_tamed = data['neutral_mean']
            self.nn_base = data['base']['nn_mean']
            self.nn_tamed = data['nn_mean']
        if self.raw_mode:
            self.timing = data['timing_iterations_mean'] * TIME_PER_ITER_TAME / 60  # divide by 60 for seconds -> minutes conversion

    def __quantiles_str(self):
        print_order = [self.forget_base, self.forget_tamed, self.forget_ref_base, self.forget_ref_tamed, self.remember_base, self.remember_tamed, self.neutral_base, self.neutral_tamed, self.nn_base, self.nn_tamed]
        print_str = f"{self.n_images}&"
        for idx, cur_num in enumerate(print_order):
            cur_num = rel_dist2likelihood_qunatile(cur_num, return_torch=False)
            print_str += self.__get_num_by_precision(cur_num)

            if idx != len(print_order) - 1:
                print_str += "&"
            else:
                print_str += r"\tabularnewline"
        return print_str

    def __quantile_drop_str(self):
        print_order = [compute_quantile_drop(rel_dist2likelihood_qunatile(theta_b, return_torch=False), rel_dist2likelihood_qunatile(theta_t, return_torch=False)) for theta_b, theta_t in
                       [(self.forget_base, self.forget_tamed),
                        (self.forget_ref_base, self.forget_ref_tamed),
                        (self.remember_base, self.remember_tamed),
                        (self.neutral_base, self.neutral_tamed),
                        (self.nn_base, self.nn_tamed)]]
        print_str = f"{self.n_images}&" + r"\greencheck&"
        for idx, cur_num in enumerate(print_order):
            print_str += self.__get_num_by_precision(cur_num)

            if idx != len(print_order) - 1:
                print_str += "&"
            else:
                print_str += r"\tabularnewline"
        return print_str

    def __str__(self) -> str:
        if self.raw_mode:
            return self.__raw_table_str()
        if self.qunatiles_mode:
            if self.quantiles_drop:
                return self.__quantile_drop_str()
            return self.__quantiles_str()
        # print_order = [self.forget_base, self.forget_tamed, self.forget_ref, self.remember, self.neutral, self.nn]
        print_order = [self.forget_ref, self.remember, self.neutral, self.nn]
        print_str = f"{self.n_images}&" + r"\multicolumn{1}{c}{\greencheck}&"
        for idx, cur_num in enumerate(print_order):
            print_str += self.__get_num_by_precision(cur_num)

            if idx != len(print_order) - 1:
                print_str += "&"
            else:
                print_str += r"\tabularnewline"
        return print_str

    def verbose_str(self) -> str:
        print_str = f"forget: base: {self.forget_base:.{self.precision}}, tamed: {self.forget_tamed:.{self.precision}}, FF: {self.forget_diff:.{self.precision}}\n"
        return print_str + self.__str__()

    def __get_num_by_precision(self, cur_num) -> str:
        return_str = ''
        if abs(cur_num) >= (10 ** (-1 * self.precision)) / 2:
            # round up from half decimal, so numbers in [1/2 * 10{-(precision + 1)}, infinity)
            # e.g. precision=2 this is [0.005, infinity)
            return_str += f"${cur_num:.{self.precision}f}$"
        else:
            # find the closest power of 10 that is greater.
            # e.g. for 0.001 we will return < 10^{-2}
            cur_precision = self.precision + 1
            while True:
                if abs(cur_num) > (10 ** (-1 * cur_precision)):
                    return_str += r"$<10^{-" + str(cur_precision - 1) + "}$"
                    break
                cur_precision += 1
        return return_str

    def __raw_table_str(self) -> str:
        print_str = f"{self.n_images}&"
        forget_values_order = [(self.forget_base, self.forget_tamed),
                               (self.forget_ref_base, self.forget_ref_tamed),
                               (self.remember_base, self.remember_tamed),
                               (self.neutral_base, self.neutral_tamed), (self.nn_base,self.nn_tamed)]
        for base_num, tamed_num in forget_values_order:
            base_num, tamed_num = rel_dist2likelihood_qunatile(base_num, return_torch=False), rel_dist2likelihood_qunatile(tamed_num, return_torch=False)

            print_str += self.__get_num_by_precision(base_num) + "&"
            print_str += self.__get_num_by_precision(tamed_num) + "&"
            print_str += self.__get_num_by_precision(compute_quantile_drop(base_num, tamed_num)) + "&"
        print_str += f"{self.timing:.1f}" + r"\tabularnewline"

        return print_str


def print_new_table(input_files_dir: str, save_path='', verbose=False, **args):
    n_images = [1, 4, 8, 15]
    assert os.path.isdir(input_files_dir) and all([os.path.isfile(f"{input_files_dir}/{n}.json") for n in n_images])
    rows = []
    for n in n_images:
        rows.append(TableRow(n, f"{input_files_dir}/{n}.json", **args))
    if save_path:
        with open(save_path, "w") as out_file:
            out_file.writelines([str(row) + "\n" for row in rows])
    for row in rows:
        print(row.verbose_str() if verbose else row)


@torch.no_grad()
def get_base_model_likelihood_on_test_identities():
    args = get_base_model_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, training=False)
    out_dir = "models/baseline/continue_celeba/distribution_stats/valid_partial_10000/test_identities_quantiles"
    os.makedirs(out_dir, exist_ok=True)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    data_sources = {}
    args.data_split = 'train'
    for identity in TEST_IDENTITIES:
        args.forget_size = 15
        args.forget_identity = identity
        cur_ds = args2dataset(args, "forget", transform)
        data_sources[str(identity)] = cur_ds
    raw_nll = get_model_nll_on_multiple_data_sources(model, device, data_sources, n_bins=2 ** args.n_bits)
    with open("models/baseline/continue_celeba/distribution_stats/valid_partial_10000/distribution.json") as base_json:
        base_dist = json.load(base_json)
        mean, sigma = base_dist["nll"]["mean"], base_dist["nll"]["std"]
    rel_dist = {ds_name: [(nll - mean) / sigma for nll in nll_vals] for ds_name, nll_vals in raw_nll.items()}
    quantiles_dict = {ds_name: [rel_dist2likelihood_qunatile(cur, return_torch=False) for cur in rel_dist] for ds_name, rel_dist in rel_dist.items()}
    save_dict_as_json(quantiles_dict, f"{out_dir}/quantiles.json")


def get_timing(base_dir, out_dir=''):
    n_images = [1, 4, 8, 15]
    # for delta in deltas:
    time_values = {n: [] for n in n_images}
    files = glob(f"{base_dir}/**/wandb/latest-run/files/output.log")
    for f in files:
        with open(f, "r") as cur_file:
            lines = [line for line in cur_file.readlines() if 'INFO:root:breaking' in line]
            assert len(lines) == 1
        cur_n_images = int(re.match(r'.*forget_identity_.*/(\d+)_image_id.*', f).group(1))
        break_num = int(re.match(r'INFO:root:breaking after (\d+) iterations.*', lines[0]).group(1))
        assert cur_n_images in time_values
        time_values[cur_n_images].append(break_num)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out = {}
    for n in n_images:
        cur_out = {'timing_iterations': time_values[n],
               'timing_iterations_mean': sum(time_values[n]) / len(time_values[n])}
        if out_dir:
            save_dict_as_json(cur_out, f"{out_dir}/{n}.json")
        out[n] = cur_out
    return out


if __name__ == '__main__':
    set_all_seeds(seed=37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



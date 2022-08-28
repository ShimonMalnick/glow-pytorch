import json
import math
import re
from time import time
import logging
import numpy as np
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, CELEBA_ROOT, \
    compute_dataset_bpd, get_default_forget_transform, np_gaussian_pdf, kl_div_univariate_gaussian, args2dataset, \
    BASELINE_MODEL_PATH, nll_to_sigma_normalized
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List, Union
from datasets import CelebAPartial
from easydict import EasyDict as edict
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
        plt.plot([], [], ' ', label=fr"$\mu_\theta={mu:.1f}$")
        plt.plot([], [], ' ', label=fr"$\sigma_\theta={std:.1f}$")
    plt.grid(False)
    plt.xlabel(r'$-\logP_X^\theta(x)$')
    plt.ylabel('Density')
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


def compute_model_distribution(exp_dir, args=None, partial=-1, **kwargs):
    if args is None:
        args = get_args(forget=True)
    if 'baseline' in kwargs:
        args.ckpt_path = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/baseline/continue_celeba/model_090001_single.pt"
    else:
        args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ["valid", "train"]:
        save_dir = f"{exp_dir}/distribution_stats/{split}"
        ds = CelebA(root=CELEBA_ROOT, target_type='identity', split=split,
                    transform=get_default_forget_transform(args.img_size, args.n_bits))
        if partial > 0:
            save_dir += f"_partial_{partial}"
            indices = torch.randperm(len(ds))[:partial]
            ds = Subset(ds, indices)
        compute_ds_distribution(256, save_dir, training=False, args=args, device=device, ds=ds, **kwargs)


def plot_2_histograms_distributions(model_dist: str, baseline_hist: str, save_path: str):
    model_dist = torch.load(model_dist).numpy()
    baseline_dist = torch.load(baseline_hist).numpy()
    from pylab import cm
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    colors = cm.get_cmap('tab20')
    plt.figure(figsize=(6, 4))
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
             label=r"$\mathcal{N}(\theta_F)$" + "\n" + rf"$\mu={model_mu:.0f}$" + "\n" + rf"$\sigma={model_sigma:.0f}$")
    baseline_mu, baseline_sigma = np.mean(baseline_dist), np.std(baseline_dist)
    plt.plot(x_vals, np_gaussian_pdf(x_vals, baseline_mu, baseline_sigma), color=colors(5),
             label=r"$\mathcal{N}(\theta_B)$" + "\n" + rf"$\mu={baseline_mu:.0f}$" + "\n" + rf"$\sigma={baseline_sigma:.0f}$")
    wasserstein_distance = (model_mu - baseline_mu) ** 2 + (model_sigma - baseline_sigma) ** 2
    kldiv_base_finetuned = kl_div_univariate_gaussian(baseline_mu, baseline_sigma, model_mu, model_sigma)
    kldiv_finetuned_base = kl_div_univariate_gaussian(model_mu, model_sigma, baseline_mu, baseline_sigma)
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


def full_experiment_evaluation(exp_dir: str, args, **kwargs):
    compute_model_distribution(exp_dir, args, **kwargs)
    baseline_base = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/baseline/continue_celeba/distribution_stats"
    evaluate_forget_score(exp_dir, **kwargs)
    for split in ['train', 'valid']:
        plot_2_histograms_distributions(f"{exp_dir}/distribution_stats/{split}_partial_10000/nll_distribution.pt",
                                        f"{baseline_base}/{split}_partial_10000/nll_distribution.pt",
                                        f"{exp_dir}/distribution_stats/{split}_partial_10000/nll_distribution.svg")


def nll_to_dict(nll_tensor, rounding=2) -> Dict:
    out = {i: round(nll_tensor[i].item(), rounding) for i in range(nll_tensor.shape[0])}
    out['mean'] = round(nll_tensor.mean().item(), rounding)
    out['std'] = round(nll_tensor.std().item(), rounding)
    return out


def compute_baseline_forget_stats(reps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/baseline/continue_celeba/args.json", "r") \
            as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = BASELINE_MODEL_PATH
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
                      "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/baseline/continue_celeba"
                      "/forget_stats/results.json")


def get_model_nll_on_multiple_data_sources(model,
                                           device,
                                           dsets_tuples: List[Tuple[str, Union[CelebAPartial, torch.Tensor]]],
                                           reps=10,
                                           n_bins=32) -> List[Tuple[str, torch.Tensor]]:
    names, data_sources = zip(*dsets_tuples)
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
    return list(zip(names, nlls))


def compare_forget_values(exp_dir, reps=10, split='valid', partial=10000):
    assert split in ['train', 'valid']
    with open(f"{exp_dir}/args.json", "r") as args_f:
        args = edict(json.load(args_f))
    args.ckpt_path = f"{exp_dir}/checkpoints/model_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    transform = get_default_forget_transform(args.img_size, args.n_bits)
    forget_ds = args2dataset(args, "forget", transform)
    trained_images = torch.stack([forget_ds[i][0] for i in range(args.forget_size)]).to(device)
    data_sources = [("trained", trained_images)]
    if args.forget_size < len(forget_ds):
        untrained_forget_images = torch.stack([forget_ds[i][0] for i in range(args.forget_size, len(forget_ds))]).to(device)
        data_sources.append(("untrained", untrained_forget_images))
    args.forget_images = "/a/home/cc/students/cs/malnick/thesis/datasets/celebA_frequent_identities/1_second/train" \
                         "/images"
    same_id_ref_ds = args2dataset(args, "forget", transform)
    data_sources.append(("same_id_ref", same_id_ref_ds))
    nll_tuples = get_model_nll_on_multiple_data_sources(model, device, data_sources, reps=reps, n_bins=2 ** args.n_bits)
    nll_dict = {ds_name: nll_to_dict(nll_tensor) for ds_name, nll_tensor in nll_tuples}
    save_dict_as_json(nll_dict, f"{exp_dir}/distribution_stats/forget_results.json")
    partial_suffix = f"_partial_{partial}" if partial > 0 else ""
    with open(f"{exp_dir}/distribution_stats/{split}{partial_suffix}/distribution.json", "r") as f:
        trained_model_distribution = json.load(f)['nll']
    trained_mean, trained_std = trained_model_distribution['mean'], trained_model_distribution['std']
    sigma_normalized_results = {ds_name + "_mean": nll_to_sigma_normalized(nll_tensor.mean(), trained_mean, trained_std)
                                for ds_name, nll_tensor in nll_tuples}
    trained_images_nlls = nll_tuples[0][1].tolist()
    sigma_normalized_results["trained_thresholds_values"] = \
        {i: nll_to_sigma_normalized(trained_images_nlls[i], trained_mean, trained_std)
         for i in range(len(trained_images_nlls))}
    passed_thresh = True
    for num in sigma_normalized_results["trained_thresholds_values"].values():
        if num < args.forget_thresh:
            passed_thresh = False
            break
    sigma_normalized_results["passed_threshold"] = passed_thresh
    save_dict_as_json(sigma_normalized_results, f"{exp_dir}/distribution_stats/{split}{partial_suffix}/forget_info.json")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    exp_dirs = glob("/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_kl_paper/*")
    for exp_dir in exp_dirs:
        logging.info(f"Comparing forget values in {exp_dir}")
        compare_forget_values(exp_dir)

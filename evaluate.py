import json
import math
import re
from time import time
import logging
import numpy as np
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, CELEBA_ROOT, \
    compute_dataset_bpd, get_default_forget_transform, compute_dataloader_bpd
import os
from forget import args2dataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List
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
    Compute data for a plot of similarity (x-axis) vs bpd (y-axis) of different identities, for the given models the
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


def get_celeba_bpd_distribution(batch_size, save_dir: str = 'outputs/celeba_stats/bpd_distribution', training=False):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args(forget=True)
    model: Glow = load_model(args, device, training=training)
    bpds = None
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
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        cur_nll = - (log_p + logdet)
        assert cur_nll.nelement() == batch_size and cur_nll.ndim == 1
        cur_bpd = cur_nll / (math.log(2) * M) + math.log(n_bins) / math.log(2)
        if bpds is None:
            bpds = cur_bpd.detach().cpu()
        else:
            bpds = torch.cat((bpds, cur_bpd.detach().cpu()))
        print(f"Finished {i}/{len(dl)} batches")

    torch.save(bpds, f"{save_dir}/bpd_distribution.pt")

    data = {'mean': torch.mean(bpds).item(),
            'std': torch.std(bpds).item(),
            'min': torch.min(bpds).item(),
            'max': torch.max(bpds).item(),
            'median': torch.median(bpds).item()}
    save_dict_as_json(data, f"{save_dir}/bpd_distribution.json")
    plt.hist(bpds.numpy(), bins=200)
    plt.title('BPD distribution histogram')
    plt.savefig(f"{save_dir}/bpd_distribution_hist.png")


if __name__ == '__main__':
    # wandb.init(project='tmp-debug', name=f'hist_{torch.randint(20, (1,)).item()}', entity="malnick")
    logging.getLogger().setLevel(logging.INFO)
    pass

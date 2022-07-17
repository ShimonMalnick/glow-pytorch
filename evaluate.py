import json
import math
import re
from time import time

import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, compute_bpd, quantize_image, CELEBA_ROOT
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List


def eval_model(dataloaders: Iterable[Tuple[str, DataLoader]], n_bits: int, model, device, img_size: int = 128,
               **kwargs) -> dict:
    n_bins = 2 ** n_bits
    data = {}
    start = time()
    for name, dl in dataloaders:
        cur_bpd = compute_bpd(n_bins, img_size, model, device, dl)
        data[name] = cur_bpd
        print(f'{name} done in {time() - start}')
        start = time()
    if 'save_dir' in kwargs:
        save_dir = kwargs['save_dir']
        save_dict_as_json(data, f'{save_dir}/bpd.json')
    return data


def eval_models(ckpts: Iterable[Tuple[str, str]], args: Dict, dsets: Iterable[Tuple[str, Dataset]], **kwargs):
    """
    Used for Evaluation of multiple models on the same data sets
    :param dsets: tuples of (name, ds) where ds is a dataset and name is the corresponding name of that data.
    :param ckpts: list of tuples of (ckpt, names) where ckpt is a path to a checkpoint and name is the corresponding
     model name
    :param args: arguments relevant for the models (same arguments to all models)
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    dataloaders = []
    for name, ds in dsets:
        dl = DataLoader(ds, batch_size=min(len(ds), 32), shuffle=False, num_workers=8)
        dataloaders.append((name, dl))
    start = time()
    for name, ckpt in ckpts:
        args.ckpt_path = ckpt
        model: Glow = load_model(args, device, training=False)
        model.eval()
        data[name] = eval_model(dataloaders, args.n_bits, model, device, img_size=args.img_size)
        print(f'{name} done in {time() - start}')
        start = time()
    if 'save_path' in kwargs:
        save_path = kwargs['save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_dict_as_json(data, save_path)

    return data


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
    transform = Compose([Resize((args.img_size, args.img_size)),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, args.n_bits)])
    base_ds = CelebA(root=CELEBA_ROOT, transform=transform, target_type='identity', split='train')
    with open("outputs/celeba_stats/identity2indices.json", 'r') as f:
        identity2indices = json.load(f)
    for identity in dsets_identities:
        if identity > 0:
            cur_ds = Subset(base_ds, identity2indices[str(identity)])
            dsets.append((f'identity_{identity}', cur_ds))
        else:
            dsets.append((f'celeba_train', base_ds))
    for folder in folders:
        last_model_path = f'{folder}/checkpoints/model_last.pt'
        if os.path.isfile(last_model_path):
            cur_checkpoint = last_model_path
        else:
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
            cur_checkpoint = sorted(glob(f'{folder}/checkpoints/model_*.pt'))[-1]
        args.ckpt_path = cur_checkpoint
        model: Glow = load_model(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"), training=False)
        model.eval()
        dsets: List[Tuple[str, Dataset]] = []
        transform = Compose([Resize((args.img_size, args.img_size)),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             lambda img: quantize_image(img, args.n_bits)])

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
        data = [(d[0], 1 - d[1]) for d in data] # cosine_distance = 1 - cosine_similarity

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


def get_celeba_bpd_distribution(batch_size, save_dir: str = 'outputs/celeba_stats/bpd_distribution'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args(forget=True)
    model: Glow = load_model(args, device, training=False)
    bpds = []
    transform = Compose([Resize((args.img_size, args.img_size)),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, args.n_bits)])
    ds = CelebA(root=CELEBA_ROOT, target_type='identity', split='train', transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

    M = args.img_size * args.img_size * 3
    n_bins = 2 ** args.n_bits
    for batch in dl:
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            log_p, logdet, _ = model(x)
        assert x.shape[0] == batch_size
        cur_nll = - torch.sum(log_p + logdet).item() / x.shape[0]
        cur_bpd = (cur_nll + (M * math.log(n_bins))) / (math.log(2) * M)
        bpds.append(cur_bpd)

    bpds = np.array(bpds)
    np.save(f"{save_dir}/bpd_distribution.npy", bpds)

    data = {'mean': np.mean(bpds),
            'std': np.std(bpds),
            'min': np.min(bpds),
            'max': np.max(bpds),
            'median': np.median(bpds)}
    save_dict_as_json(data, f"{save_dir}/bpd_distribution.json")


if __name__ == '__main__':
    get_celeba_bpd_distribution(32)
    # folders = glob("experiments/forget/*thresh_1e4*")
    # folder = ['experiments/forget/all_images_thresh_1e3']
    # run_eval_on_models(folder, [1, 13, 15], save_path=f"{folder[0]}/bpd.json")

    # folders = list(set(folders) - set(glob("experiments/forget/*baseline*")))
    # run_eval_on_models(folders, [1, 4, 6, 7, 8, 12, 13, 14, 15], save_path='outputs/forget_bpd/forget_thresh_1e4.json')
    # run_eval_on_models(folders, [1, 13, 15], save_path='outputs/forget_bpd/forget_thresh_1e4.json')
    # compute_similarity_vs_bpd(folders, step=20)
    # base_path = "/home/yandex/AMNLP2021/malnick/glow_repos/glow-pytorch-rosinality/experiments/forget/1_image_thresh_1e4"
    # for p in folders:
    #     plot_similarity_vs_bpd(f'{p}/bpd_vs_similarity.png', f'{p}/bpd_vs_similarity.json')
    #     plot_similarity_vs_bpd(f'{p}/bpd_vs_similarity_wo_outliers.png', f'{p}/bpd_vs_similarity.json', outliers_thresh=50)
    # plot_similarity_vs_bpd(f"{base_path}/bpd_vs_similarity.png", f"{base_path}/bpd_vs_similarity.json")

import json
from torchvision.datasets import CelebA
from utils import get_args, save_dict_as_json, load_model, compute_bpd, quantize_image, CELEBA_NUM_IDENTITIES, \
    CELEBA_ROOT
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from easydict import EasyDict
from forget import args2dset_params, get_partial_dataset
from glob import glob
from model import Glow
from typing import Iterable, Tuple, Dict, List


def eval_model(dataloaders: Iterable[Tuple[str, DataLoader]], n_bits: int, model, device, img_size: int = 128,
               **kwargs) -> dict:
    n_bins = 2 ** n_bits
    data = {}
    for name, dl in dataloaders:
        cur_bpd = compute_bpd(n_bins, img_size, model, device, dl)
        data[name] = cur_bpd
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

    for name, ckpt in ckpts:
        args.ckpt_path = ckpt
        model: Glow = load_model(args, device, training=False)
        model.eval()
        data[name] = eval_model(dataloaders, args.n_bits, model, device, img_size=args.img_size)
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
    params_dict = EasyDict({})
    transform = Compose([Resize((args.img_size, args.img_size)),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, args.n_bits)])
    for identity in dsets_identities:
        if identity < 1:
            # used when celeba is required with all identities and images (according to the split)
            if args.data_split:
                ds = get_partial_dataset(transform, split=args.data_split)
            else:
                ds = get_partial_dataset(transform)
        else:
            params_dict.forget_identity = identity
            cur_params = args2dset_params(params_dict, 'forget')
            ds = get_partial_dataset(transform=transform, **cur_params)
        dsets.append((f'identity_{identity}', ds))
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

        params_dict = EasyDict({})
        # base_ds = CelebA(root=CELEBA_ROOT, transform=transform, target_type='identity', split='train')
        # with open("outputs/celeba_stats/identity2indices.json", 'r') as f:
        #     identity2indices = json.load(f)
        for identity in relevant_ids:
            # cur_ds = Subset(base_ds, identity2indices[str(identity)])
            params_dict.forget_identity = identity
            cur_params = args2dset_params(params_dict, 'forget')
            ds = get_partial_dataset(transform=transform, **cur_params)
            dsets.append((f'identity_{identity}', ds))
        data = eval_models([(os.path.basename(os.path.normpath(folder)), cur_checkpoint)], args, dsets, **kwargs)
        save_path = f'{folder}/bpd_vs_similarity.json'
        save_dict_as_json(data, save_path)


if __name__ == '__main__':
    # folders = glob("experiments/forget/*thresh*")
    # folders = list(set(folders) - set(glob("experiments/forget/*baseline*")))
    # run_eval_on_models(folders, [1, 4, 6, 7, 8, 12, 13, 14, 15], save_path='outputs/forget_bpd/forget_thresh_1e4.json')
    folders = ["experiments/forget/all_images_thresh_1e4"]
    compute_similarity_vs_bpd(folders, step=10)
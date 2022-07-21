import json
import logging
import math
import os
from glob import glob
from typing import List, Tuple, Dict, Union
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Subset
from utils import get_dataset, create_horizontal_bar_plot, CELEBA_ROOT, CELEBA_NUM_IDENTITIES, compute_cosine_similarity
from time import time
from multiprocessing import Pool
from collections import Counter
from functools import reduce
from utils import load_arcface, load_arcface_transform, save_dict_as_json
import shutil
from forget import get_partial_dataset
import torchvision.datasets as vision_dsets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_celeba_stats(split='train', out_dir='outputs/celeba_stats'):
    file_name = os.path.join(out_dir, f'identities_{split}.pt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(file_name):

        begin = time()
        ds = get_dataset(CELEBA_ROOT, 128, data_split=split)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
        cur = time()
        print("building dataset took: ", round(cur - begin, 2), " seconds")
        all_identities = []

        for _, labels in dl:
            all_identities.append(labels)
        all_identities = torch.cat(all_identities)
        print("all_ids_shape: ", all_identities.shape)
        print("iterating through dataset took: ", round(time() - cur, 2), " seconds")
        torch.save(all_identities, file_name)
    else:
        all_identities = torch.load(file_name)

    # slicing to remove 0 since identities are labeled {1,2, ..., num_identities}
    hist_tensor = torch.bincount(all_identities, minlength=CELEBA_NUM_IDENTITIES)[1:].float()

    print(hist_tensor[:50])
    non_zero_elements = hist_tensor[hist_tensor.nonzero()]
    data = {'split': split,
            'ds_size': all_identities.nelement(),
            'num_identities': torch.unique(all_identities).nelement(),
            "min_freq": round(torch.min(hist_tensor).item(), 2),
            "max_freq": round(torch.max(hist_tensor).item(), 2),
            "mean_freq": round(torch.mean(hist_tensor).item(), 2),
            "std_freq": round(torch.std(hist_tensor).item(), 2),
            "median_freq": round(torch.median(hist_tensor).item(), 2),
            "num_zeros": hist_tensor.nelement() - non_zero_elements.nelement(),
            "non_zero_mean": round(non_zero_elements.mean().item(), 2),
            "non_zero_std": round(non_zero_elements.std().item(), 2),
            "non_zero_median": round(torch.median(non_zero_elements).item(), 2)}
    print("min id: ", torch.min(all_identities))
    print("max id: ", torch.max(all_identities))
    with open(os.path.join(out_dir, f'stats_{split}.json'), 'w') as f:
        json.dump(data, f, indent=4)
    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(CELEBA_NUM_IDENTITIES), hist_tensor.numpy())
    plt.savefig(os.path.join(out_dir, f'hist_{split}.png'))


def save_images_chosen_identities(ids: List[int], save_dir: str, split: str = 'train'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images_root = os.path.join(CELEBA_ROOT, "celeba", "img_align_celeba")
    ds = get_dataset(CELEBA_ROOT, 128, data_split=split)
    all_identities = torch.load(os.path.join('outputs/celeba_stats', f'identities_{split}.pt'))
    for identity in ids:
        cur_dir = os.path.join(save_dir, str(identity), split, "images")  # images added just to use ImageFolder dataset
        os.makedirs(cur_dir, exist_ok=True)
        ids_tensors = (all_identities == identity).nonzero(as_tuple=True)[0]
        print(f"for split {split} and identity {identity} got: {ids_tensors.shape} images")
        for i, id_tensor in enumerate(ids_tensors):
            cur_file_name = ds.filename[id_tensor]
            _, cur_label = ds[id_tensor]
            assert cur_label.item() == identity
            shutil.copy2(os.path.join(images_root, cur_file_name), cur_dir)


def parse_line(line):
    line = line.split(' ')[1:]
    line = [part for part in line if part]
    return Counter({i: int(line[i]) for i in range(len(line)) if line[i] == '1'})


def get_celeba_attributes_stats(
        attr_file_path: str = '/home/yandex/AMNLP2021/malnick/datasets/celebA/celeba/list_attr_celeba.txt',
        save_stats_path='outputs/celeba_stats/attributes_stats.json'):
    start_parse = time()
    with open(attr_file_path, 'r') as f:
        num_images = int(f.readline().strip())
        attributes = f.readline().strip().split(' ')
        attributes_dict = {attribute: 0 for attribute in attributes}
        attributes_map = {i: attribute for i, attribute in enumerate(attributes)}
        lines = []
        for i in range(num_images):
            lines.append(f.readline().strip())
        with Pool(16) as p:
            map_result = p.map(parse_line, lines)
        reduced = reduce(lambda d1, d2: d1 + d2, map_result)
        for k in reduced:
            attributes_dict[attributes_map[k]] = reduced[k]
    end_parse = time()
    print("Parsing all attributes took: ", round(end_parse - start_parse, 2), " seconds")
    if save_stats_path:
        with open(save_stats_path, 'w') as f:
            json.dump(attributes_dict, f, indent=4)
    save_time = time()
    print("saving took ", round(save_time - end_parse, 2), " seconds")
    create_horizontal_bar_plot(attributes_dict, 'outputs/celeba_stats/celeba_attributes_stats.png',
                               title='CelebA Binary Atrributes Count (out of {} images)'.format(num_images))
    print("plotting took ", round(time() - save_time, 2), " seconds")


def similarity_to_distance(similarity, mean=False):
    dist = 1 - similarity
    if mean:
        return dist.mean()
    return dist


def images_to_similarity(folders: List[Tuple[str, str]], save_path='outputs/celeba_stats/similarity.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls = load_arcface(device=device)
    t = load_arcface_transform()
    scores = {}
    for i, (name, folder) in enumerate(folders):
        paths = glob(f"{folder}/*")
        tensors = torch.stack([t(Image.open(path)) for path in paths]).to(device)
        cur_embeddings = cls(tensors)
        for j in range(i, len(folders)):
            name_j = folders[j][0]
            folder_j = folders[j][1]
            paths_j = glob(f"{folder_j}/*")
            tensors_j = torch.stack([t(Image.open(path)) for path in paths_j]).to(device)
            embeddings_j = cls(tensors_j)
            similarity = compute_cosine_similarity(cur_embeddings, embeddings_j, mean=True)
            scores[f"{name},{name_j}"] = similarity.item()

    save_dict_as_json(save_dict=scores, save_path=save_path)
    return scores


def get_identity2identities_sim(chosen_images: List[str],
                                indices_json_path: str = 'outputs/celeba_stats/identity2indices.json',
                                save_path: str = 'outputs/celeba_stats/1_first_similarities.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arcface = load_arcface(device=device)
    transform = load_arcface_transform()

    # compute embeddings of chosen images
    chosen_images = torch.stack([transform(Image.open(path)) for path in chosen_images]).to(device)
    chosen_embeddings = arcface(chosen_images)

    identities2indices: Dict[str, List[int]]
    with open(indices_json_path, 'r') as f:
        identities2indices = json.load(f)
    base_ds = vision_dsets.CelebA(root=CELEBA_ROOT, download=False, transform=transform, split='train',
                                  target_type='identity')
    similarities = {}
    for idx, (identity, indices) in enumerate(identities2indices.items(), 1):
        cur_ds = Subset(base_ds, indices)
        cur_id = int(identity)
        assert all([cur_ds[i][1].item() == cur_id for i in range(len(cur_ds))])
        cur_images = torch.stack([cur_ds[i][0] for i in range(len(cur_ds))]).to(device)
        cur_embeddings = arcface(cur_images)
        cur_sim = compute_cosine_similarity(chosen_embeddings, cur_embeddings, mean=True)
        similarities[cur_id] = cur_sim.item()
        logging.info(f"Finished {idx}/{len(identities2indices)}")

    similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}
    save_dict_as_json(similarities, save_path)


def get_identity2identities_similarity(identity: int = None, images: List[str] = None):
    start = time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arcface = load_arcface(device=device)
    transform = load_arcface_transform()
    if identity is not None:
        base_id_ds = get_partial_dataset(transform, include_only_identities=[identity])
        rest_of_ids_ds = get_partial_dataset(transform, exclude_identities=[identity])
    elif images is not None:
        base_id_ds = get_partial_dataset(transform, include_only_images=images)
        rest_of_ids_ds = get_partial_dataset(transform, exclude_images=images)
    else:
        raise ValueError("Either identity or images must be specified")
    base_id_tensors = torch.stack([base_id_ds[i][0].to(device) for i in range(len(base_id_ds))])
    base_id_embeddings = arcface(base_id_tensors)
    rest_dl = DataLoader(rest_of_ids_ds, batch_size=256, shuffle=False)
    ids_similarities = {i: [] for i in range(1, CELEBA_NUM_IDENTITIES + 1) if i != identity}
    for i, (x, y) in enumerate(rest_dl):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            embeddings = arcface(x)  # of shape (128, E)
        # nest line is of shape (#base_id images, 128)
        similarity = compute_cosine_similarity(base_id_embeddings, embeddings, mean=False)
        similarity = torch.mean(similarity, dim=0)  # shape (128)
        batch_ids = torch.unique(y).tolist()
        for id_num in batch_ids:
            ids_similarities[id_num].append(similarity[y == id_num].detach().cpu())
        print(f"finished {i} batches in {round(time() - start, 2)} seconds")

    outputs = {k: torch.cat(v).mean().item() for k, v in ids_similarities.items() if len(v) > 0}
    torch.save(outputs, f'outputs/celeba_stats/{identity}_similarities.pt')
    print("finished all in ", round(time() - start, 2), " seconds")
    outputs = {k: v for k, v in sorted(outputs.items(), key=lambda item: item[1])}
    save_dict_as_json(outputs, f'outputs/celeba_stats/{identity}_similarities.json')


def compute_celeba_identity2indices(save_path='outputs/celeba_stats/identity2indices.json'):
    ds = vision_dsets.CelebA(root=CELEBA_ROOT, target_type='identity', split='train', transform=ToTensor())
    identity2indices = {}
    start = time()
    for i in range(len(ds)):
        _, y = ds[i]
        identity: int = y.item()
        if (i + 1) % 1000 == 0:
            print(f"finished {i} images in {round(time() - start, 2)} seconds")
            start = time()
        if identity not in identity2indices:
            identity2indices[identity] = []
        identity2indices[identity].append(i)
    save_dict_as_json(identity2indices, save_path)


def plot_identity_neighbors(neighbors_index: List[int], chosen_id: Union[int, str] = "1_first",
                            save_path='outputs/identity_1_first/neighbors.png',
                            similaririty_path='outputs/celeba_stats/1_first_similarities.json'):
    """
    Given an identity from celeba, plot neighbors of that identities according to given distance indices
    :param similaririty_path:
    :param save_path: save path of the plot.
    :param neighbors_index: the indices of identities, meaning if idx = 1 we will choose the nearest neighbor of id1,
    if idx = -1 the furthest identity from id1, etc.
    :param chosen_id: the identity to plot neighbors
    """
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(chosen_id, str):
        chosen_id_dir = f"/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/frequent_identities/{chosen_id}/train/images"
        assert os.path.isdir(chosen_id_dir)
    similarities: List[Tuple[int, float]]
    identity2indices: Dict[str, List[int]]
    with open("outputs/celeba_stats/identity2indices.json", "r") as f:
        identity2indices = json.load(f)
    with open(similaririty_path, "r") as f:
        similarities = list(reversed([(int(k), v) for k, v in json.load(f).items()]))
    assert all([abs(i) < len(similarities) for i in neighbors_index]), "found neighbors index out of range"
    n_neighbors = len(neighbors_index)
    num_images = n_neighbors + 1  # added one for the chosen id
    n_rows = math.floor(math.sqrt(num_images))
    n_cols = math.ceil(num_images / n_rows)
    print(f"n_rows: {n_rows}, n_cols: {n_cols}")
    fig = plt.figure(figsize=(15, 15))
    plt.title(f"nearest neighbors of identity {chosen_id} out of {len(similarities)} neighbors".title())
    plt.axis('off')
    # load dataset
    celeba_ds = vision_dsets.CelebA(root=CELEBA_ROOT, target_type='identity', split='train')
    print(f"Loaded CelebA", flush=True)

    def identity2image(identity: int) -> Image:
        assert identity2indices[str(identity)], "no images found for this identity"
        lst = identity2indices[str(identity)]
        # print(lst)
        idx = random.choice(lst)
        image, label = celeba_ds[idx]
        assert label == identity, f"label: {label} and identity: {identity} do not match"
        return image

    def absolute_index(idx, arr_len):
        return idx if idx >= 0 else arr_len + idx

    # plot identity
    if isinstance(chosen_id, int):
        id_image = identity2image(chosen_id)
    elif isinstance(chosen_id, str):
        id_image = Image.open(glob(f"{chosen_id_dir}/*")[0])
    else:
        raise ValueError(f"chosen_id must be int or str, got {type(chosen_id)}")
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.imshow(id_image)
    ax.title.set_text(f"id: {chosen_id}")
    ax.axis('off')

    # plot neighbors
    for i in range(len(neighbors_index)):
        cur_identity, cur_similarity = similarities[neighbors_index[i]]
        cur_image = identity2image(cur_identity)
        cur_ax = fig.add_subplot(n_rows, n_cols, i + 2)
        cur_ax.imshow(cur_image)
        # cur_ax.title.set_text(f"N={neighbors_index[i]}:{absolute_index(neighbors_index[i], len(similarities))}")
        cur_ax.title.set_text(f"N={neighbors_index[i]}")
        cur_ax.axis('off')
        print("rendered image ", i)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    for i in range(3, 7):
        plot_identity_neighbors([1, 2, 3, 4, 5, 10, 20, -20, -10, -5, -4, -3, -2, -1], save_path=f"outputs/identity_1_first/{i}neighbors.png")
    exit(1)
    bpds = torch.load("outputs/baseline_stats/all_images_bpd.pt").cpu()
    data = {"mean": bpds.mean().item(), "std": bpds.std().item(), "min": bpds.min().item(), "max": bpds.max().item(),
            "median": torch.median(bpds).item()}
    save_dict_as_json(data, "outputs/baseline_stats/all_images_bpd.json")

    pass
    # logging.getLogger().setLevel(logging.INFO)
    # images = glob("/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/frequent_identities/1_first/train/images/*")

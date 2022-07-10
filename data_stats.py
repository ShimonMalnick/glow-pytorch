import json
import os
from glob import glob
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_dataset, create_horizontal_bar_plot, CELEBA_ROOT, CELEBA_NUM_IDENTITIES, compute_cosine_similarity
from time import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import Counter
from functools import reduce
from utils import load_arcface, load_arcface_transform, save_dict_as_json
import shutil
from forget import get_partial_dataset
import torchvision.datasets as vision_dsets
from torchvision.transforms import ToTensor


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


def save_images_chosen_identities(ids: List[int], save_dir: str, split: str='train'):
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


def get_celeba_attributes_stats(attr_file_path: str='/home/yandex/AMNLP2021/malnick/datasets/celebA/celeba/list_attr_celeba.txt',
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


def get_identity2identities_similarity(identity: int):
    start = time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arcface = load_arcface(device=device)
    transform = load_arcface_transform()
    base_id_ds = get_partial_dataset(transform, include_only_identities=[identity])

    base_id_tensors = torch.stack([base_id_ds[i][0].to(device) for i in range(len(base_id_ds))])
    base_id_embeddings = arcface(base_id_tensors)
    rest_of_ids_ds = get_partial_dataset(transform, exclude_identities=[identity])
    rest_dl = DataLoader(rest_of_ids_ds, batch_size=128, shuffle=False)
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
    start = time()
    ds = vision_dsets.CelebA(root=CELEBA_ROOT, target_type='identity', split='train', transform=ToTensor())
    dl = DataLoader(ds, batch_size=1024, shuffle=False)
    identity2indices = {}
    for _, y in dl:
        for i in range(len(y)):
            identity = y[i].item()
            if identity not in identity2indices:
                identity2indices[identity] = []
            identity2indices[identity].append(i)
    save_dict_as_json(identity2indices, save_path)
    print(f"took {round(time() - start, 2)} seconds")


if __name__ == '__main__':
    compute_celeba_identity2indices()

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
from utils import get_dataset, create_horizontal_bar_plot, CELEBA_ROOT, CELEBA_NUM_IDENTITIES, \
    compute_cosine_similarity, get_partial_dataset, TEST_IDENTITIES
from time import time
from multiprocessing import Pool
from collections import Counter
from functools import reduce
from utils import load_arcface, load_arcface_transform, save_dict_as_json
import shutil
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
        attr_file_path: str = f'{CELEBA_ROOT}/celeba/list_attr_celeba.txt',
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


def copy_dir2split_dirs(dir_path: str, first_save_dir: str, second_save_dir: str, dry_run=False, num_images_first=None):
    if not dry_run and not os.path.exists(first_save_dir):
        os.makedirs(first_save_dir)
    if not dry_run and not os.path.exists(second_save_dir):
        os.makedirs(second_save_dir)
    files = os.listdir(dir_path)
    if num_images_first is not None:
        mid = num_images_first
    else:
        mid = len(files) // 2
    first, second = files[:mid], files[mid:]
    for file in first:
        if dry_run:
            print("source: ", os.path.join(dir_path, file))
            print("dest: ", os.path.join(first_save_dir, file))
        else:
            shutil.copy(os.path.join(dir_path, file), second_save_dir)
    for file in second:
        if dry_run:
            print("source: ", os.path.join(dir_path, file))
            print("dest: ", os.path.join(second_save_dir, file))
        else:
            shutil.copy(os.path.join(dir_path, file), first_save_dir)


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


def create_celeba_subset_folder(identities_file: str, num_identities: int, out_path: str):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(identities_file, "r") as in_f:
        frequent_ids = [int(line.strip()) for line in in_f.readlines()]
    chosen_ids = random.sample(frequent_ids, num_identities)
    for identity in chosen_ids:
        identity_dir = f"{out_path}/{identity}"
        os.makedirs(identity_dir)
        os.makedirs(f"{identity_dir}/train/images")
    with open("/a/home/cc/students/cs/malnick/thesis/datasets/celebA/celeba/identity_CelebA.txt", "r") as in_f:
        for line in in_f.readlines():
            line = line.strip()
            file, identity = line.split()
            identity = int(identity)
            if identity in chosen_ids:
                shutil.copy(f"/a/home/cc/students/cs/malnick/thesis/datasets/celebA/celeba/img_align_celeba/{file}",
                            f"{out_path}/{identity}/train/images/{file}")


def gather_runs_forget_statistics(runs_dir: str, out_dir: str, num_forgets: List[int] = None):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if num_forgets is None:
        num_forgets = [1, 4, 8, 15]
    data = {n: {"iter": [],
                "forget_mean": [],
                "ref_forget_identity_mean": [],
                "ref_random_mean": [],
                "baseline_forget": [],
                "baseline_ref_images": [],
                "baseline_random_ref_mean": []}
            for n in num_forgets}

    for identity in num_forgets:
        relevant_runs = glob(f"{runs_dir}/{identity}_image_id_*")
        for run in relevant_runs:
            with open(f"{run}/wandb/wandb/latest-run/files/output.log") as args_file:
                log_lines = args_file.readlines()
                for line in log_lines:
                    if 'INFO:root:breaking after' in line:
                        iter = int(line.split()[-2])
                        data[identity]["iter"].append(iter)
                        break
                else:
                    print("didn't find threshold for run: ", run)
                    with open(f"{run}/args.json", "r") as args_file:
                        args = json.load(args_file)
                        iter = args["iter"]
                        data[identity]["iter"].append(iter)
            with open(f"{run}/distribution_stats/valid_partial_10000/forget_info.json", "r") as forget_file:
                forget_info = json.load(forget_file)
            data[identity]["forget_mean"].append(forget_info["forget_mean"])
            data[identity]["ref_forget_identity_mean"].append(forget_info["ref_forget_identity_mean"])
            data[identity]["ref_random_mean"].append(forget_info["ref_random_mean"])
            data[identity]["baseline_forget"].append(forget_info["baseline"]["forget"])
            data[identity]["baseline_ref_images"].append(forget_info["baseline"]["ref_images"])
            data[identity]["baseline_random_ref_mean"].append(forget_info["baseline"]["random_ref_mean"])
    mean_data = {n: {} for n in num_forgets}
    for n in data:
        for k in list(data[n]):
            mean_data[n][k] = sum(data[n][k]) / len(data[n][k])
    with open(f"{out_dir}/forget_all_identities_statistics.json", "w") as out_file:
        json.dump(data, out_file, indent=4)
    with open(f"{out_dir}/forget_all_identities_statistics_mean.json", "w") as mean_out_file:
        json.dump(mean_data, mean_out_file, indent=4)


def get_paper_table_data(forget_json_file: str, output_file: str, avg_time_per_iter=17.28):
    with open(forget_json_file, "r") as in_f:
        data = json.load(in_f)
    time_per_iter = avg_time_per_iter
    baseline_time_per_iter = 1.93  # in seconds
    baseline_n_iters = 590000
    with open(output_file, "w") as out_f:
        for k in ["1", "4", "8", "15"]:
            dff = round(data[k]["forget_mean"], 2)
            dfb = round(data[k]["baseline_forget"], 2)
            df_tag_f = round(data[k]["ref_forget_identity_mean"], 2)
            df_tag_b = round(data[k]["baseline_ref_images"], 2)
            drf = round(data[k]["ref_random_mean"], 2)
            drb = round(data[k]["baseline_random_ref_mean"], 2)
            t_min = round(data[k]["iter"] * time_per_iter / 60, 1)
            t_against_baseline = (data[k]["iter"] * time_per_iter) / (baseline_time_per_iter * baseline_n_iters)
            t_percentage = round(t_against_baseline * 100, 2)
            cur_line = fr"$\abs{{\D_F}}={k}$&{dff}&{dfb}&{df_tag_f}&{df_tag_b}&{drf}&{drb}&{t_min}&{t_percentage}\%\tabularnewline"
            out_f.write(cur_line + "\n")


if __name__ == '__main__':
    log_num = 10
    base_dir = f"experiments/forget_all_identities_log_{log_num}"
    out_dir = f"experiments/all_identities_log_{log_num}_stats"
    gather_runs_forget_statistics(base_dir, out_dir)

    forget_file = f"experiments/all_identities_log_{log_num}_stats/forget_all_identities_statistics_mean.json"
    output_file = f"experiments/all_identities_log_{log_num}_stats/forget_all_identities_statistics_mean.tex"
    get_paper_table_data(forget_file, output_file, avg_time_per_iter=4.73)

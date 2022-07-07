import json
import os
from typing import List
from torchvision.utils import save_image
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_dataset, create_horizontal_bar_plot, CELEBA_ROOT, CELEBA_NUM_IDENTITIES
from time import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import Counter
from functools import reduce
from datasets import CelebAPartial
import shutil


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


if __name__ == '__main__':
    get_celeba_attributes_stats()

    # train_ids = torch.load(os.path.join('outputs/celeba_stats', f'identities_train.pt'))
    # train_hist = torch.bincount(train_ids, minlength=CELEBA_NUM_IDENTITIES)[1:].float()
    # save_images_chosen_identities([1, 4, 6, 7, 8, 12, 13, 14, 15], '/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/frequent_identities',
    #                               split='train')



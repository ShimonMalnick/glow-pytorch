import json
import os
from typing import List
from torchvision.utils import save_image
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_dataset
from time import time
import matplotlib.pyplot as plt


def get_celeba_stats(split='train', out_dir='outputs/celeba_stats'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'identities.pt')):

        begin = time()
        ds = get_dataset('/home/yandex/AMNLP2021/malnick/datasets/celebA', 128, split=split)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
        cur = time()
        print("building dataset took: ", round(cur - begin, 2), " seconds")
        all_identities = []

        for _, labels in dl:
            all_identities.append(labels)
        all_identities = torch.cat(all_identities)
        print("all_ids_shape: ", all_identities.shape)
        print("iterating through dataset took: ", round(time() - cur, 2), " seconds")
        torch.save(all_identities, os.path.join(out_dir, 'identities.pt'))
    else:
        all_identities = torch.load(os.path.join(out_dir, 'identities.pt'))
    num_identities = 10177
    hist_tensor = torch.bincount(all_identities, minlength=num_identities)[1:].float()
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
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(data, f, indent=4)
    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(num_identities), hist_tensor.numpy())
    plt.savefig(os.path.join(out_dir, 'hist.png'))


def save_images_chosen_identities(ids: List[int], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ds = get_dataset('/home/yandex/AMNLP2021/malnick/datasets/celebA', 128, split='train')
    all_identities = torch.load(os.path.join('outputs/celeba_stats', 'identities.pt'))
    for identity in ids:
        cur_dir = os.path.join(save_dir, str(identity))
        os.makedirs(cur_dir, exist_ok=True)
        ids_tensors = (all_identities == identity).nonzero(as_tuple=True)[0]
        print(ids_tensors.shape)
        for i, id_tensor in enumerate(ids_tensors):
            cur_img, cur_label = ds[id_tensor]
            assert cur_label.item() == identity
            save_image(cur_img, os.path.join(cur_dir, str(i) + '.png'))


if __name__ == '__main__':
    save_dir = '/home/yandex/AMNLP2021/malnick/datasets/celebA_subsets/train_set_frequent_identities'
    save_images_chosen_identities(ids=[4], save_dir=save_dir)
    # get_celeba_stats()
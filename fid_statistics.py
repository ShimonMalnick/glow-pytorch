import json
import logging
import subprocess

import torch
import os.path
import shutil

from PIL import Image
from easydict import EasyDict
from torchvision.datasets import CelebA
from utils import CELEBA_ROOT, set_all_seeds, load_model
from train import calc_z_shapes


def generate_random_celeba_subset(out_dir: str, n_samples: int = 50000):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ds = CelebA(root=CELEBA_ROOT, split='train', download=False)
    rand_indices = torch.randperm(len(ds))
    subset_indices = rand_indices[:n_samples].tolist()
    base_images_dir = os.path.join(CELEBA_ROOT, "celeba", 'img_align_celeba')
    for idx in subset_indices:
        shutil.copy(f"{base_images_dir}/{ds.filename[idx]}", out_dir)


def generate_model_subset(model_path: str, model_args_path: str, out_dir: str, n_samples: int = 50000, device=None,
                          global_count=0):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(model_args_path, 'r') as f:
        args = EasyDict(json.load(f))
    args.ckpt_path = model_path
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device=device, training=False)
    batch_size = 256
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    temp = 0.5
    batch_sizes = [batch_size] * (n_samples // batch_size)
    if n_samples % batch_size != 0:
        batch_sizes.append(n_samples % batch_size)
    with torch.no_grad():
        for iter_num, bs in enumerate(batch_sizes):
            cur_zs = []
            for shape in z_shapes:
                cur_zs.append(torch.randn(bs, *shape, device=device) * temp)
            cur_images = model.reverse(cur_zs, reconstruct=False).cpu() + 0.5  # added 0.5 for normalization
            cur_images = cur_images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            for i in range(cur_images.shape[0]):
                cur_im = Image.fromarray(cur_images[i])
                cur_im.save(os.path.join(out_dir, f"temp_{int(temp * 10)}_sample_{global_count}.png"))
                global_count += 1
            logging.info(f"Finished batch {iter_num + 1}/{len(batch_sizes)}")


def compute_fid_score(data_dir_1, data_dir_2, out_path):
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    output = subprocess.check_output(f"python -m pytorch_fid {data_dir_1} {data_dir_2} --device {device_name}",
                                     shell=True).decode("utf-8")
    with open(out_path, "w") as f:
        f.write(output)


if __name__ == '__main__':
    set_all_seeds(seed=37)
    logging.getLogger().setLevel(logging.INFO)
    celeba_images_path = "/a/home/cc/students/cs/malnick/thesis/datasets/celeba_train_random_subset"
    # model_path = "experiments/fid_scores/1_image_id_1624/model_last.pt"
    # model_args = "experiments/fid_scores/1_image_id_1624/args.json"
    out_path = "experiments/fid_scores/1_image_id_1624/generated_samples"
    # generate_model_subset(model_path, model_args, out_path, n_samples=50000)
    # print("Finished generation")
    compute_fid_score(out_path, celeba_images_path, "experiments/fid_scores/1_image_id_1624/fid_score.txt")

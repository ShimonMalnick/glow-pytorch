import json
import logging
import subprocess
from glob import glob

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


def compute_fid_score(data_dir_1: str, data_dir_2: str, out_path: str, batch_size: int = 50, device_id=None):
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
        if device_id is not None:
            device_name += f":{device_id}"
    command = f"python -m pytorch_fid {data_dir_1} {data_dir_2} --device {device_name} --batch-size {batch_size}"
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    with open(out_path, "w") as f:
        f.write(output)


if __name__ == '__main__':
    set_all_seeds(seed=37)
    logging.getLogger().setLevel(logging.INFO)
    celeba_images_path = "/a/home/cc/students/cs/malnick/thesis/datasets/celeba_train_random_subset"

    base_dir = "experiments/fid_scores"
    dirs = glob(f"{base_dir}/ablation*")
    for d in dirs:
        if os.path.isfile(f"{d}/score.txt"):
            continue
        if not os.path.isdir(f"{d}/generated_images"):
            logging.info(f"Generating images for {d}")
            generate_model_subset(f"{d}/model_last.pt", f"{d}/args.json", f"{d}/generated_images", n_samples=50000)
        compute_fid_score(celeba_images_path, f"{d}/generated_images", f"{d}/score.txt", batch_size=50)
    # generate_model_subset(baseline_ckpt, baseline_args, "experiments/ablation_samples/baseline", n_samples=100)
    # images = glob(f"{base_dir}/*15_image_id_10015/generated_images/temp_5_sample_[0-9].png")

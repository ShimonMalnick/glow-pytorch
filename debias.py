import math

from matplotlib import pyplot as plt

from utils import load_model
import torch
from train import calc_z_shapes
from easydict import EasyDict as edict
from torchvision.utils import save_image


def create_debias_images_grid(baseline_model_path: str, debaised_model_path: str, out_path):
    args = edict(
        {"n_flow": 32,
         "n_block": 4,
         'affine': False,
         "no_lu": False,
         "temp": 0.5,
         "n_samples": 4})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.ckpt_path = baseline_model_path
    baseline = load_model(args, device, training=False)
    args.ckpt_path = debaised_model_path
    debaised = load_model(args, device, training=False)
    z_shapes = calc_z_shapes(3, 128, args.n_flow, args.n_block)
    cur_zs = []
    plt.figure(figsize=(10, 10))
    f, axarr = plt.subplots(2, args.n_samples)
    f.subplots_adjust(0, 0, 1, 1)
    for i, model in enumerate([baseline,debaised]):
        for shape in z_shapes:
            cur_zs.append(torch.randn(args.n_samples, *shape).to(device) * args.temp)
        with torch.no_grad():
            model_images = model.reverse(cur_zs, reconstruct=False).cpu() + 0.5  # added 0.5 for normalization
            model_images = model_images.clamp(0, 1).permute(0, 2, 3, 1).numpy()
        for j in range(args.n_samples):
            axarr[i, j].imshow(model_images[j])
            axarr[i, j].axis('off')
    plt.grid(False)
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    baseline_path = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/baseline/continue_celeba/model_090001.pt"
    debiased_path = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_attribute/glasses_debias_alpha_1/checkpoints/model_last.pt"
    create_debias_images_grid(baseline_path, debiased_path, "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/experiments/forget_attribute/glasses_debias_alpha_1/debias_samples.png")

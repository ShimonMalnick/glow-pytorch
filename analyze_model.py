from typing import List
from PIL import Image, ImageFont, ImageDraw
import imageio
import matplotlib.pyplot as plt
import torch
from utils import get_args
from torchvision import utils
from torchvision.transforms import ToTensor, Resize, Compose
from model import Glow
from train import calc_z_shapes
import json
import numpy as np


def load_model(args, device):
    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model = torch.nn.DataParallel(model_single)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage))
    model.to(device)

    return model_single


def get_z(args, device) -> List:
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    z_sample = []
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    return z_sample


def sample_from_model(z_samples, model, out_name='samples.png', reconstruct=True):
    with torch.no_grad():
        cur_tensors = model.reverse(z_samples, reconstruct=reconstruct).cpu()
        utils.save_image(
            cur_tensors,
            out_name,
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )


def sample_different_temp(args, device):
    model = load_model(args, device)
    args.temp = 1.0
    stationary_z = get_z(args, device)

    for temp in [0.1 * i for i in range(1, 11)]:
        cur_z = [z.clone() * temp for z in stationary_z]
        sample_from_model(cur_z, model, out_name=f'samples_{round(temp, 2)}.png')


def make_interpolation_video(args, im1_path, im2_path, model, device, alpha_steps=41, alpha_max=2.0, base_path='outputs/interpolation'):
    steps = torch.linspace(0, alpha_max, alpha_steps)
    im1, im2 = Image.open(im1_path), Image.open(im2_path)
    image_transform = Compose([ToTensor(), Resize((args.img_size, args.img_size))])
    tens_1, tens_2 = image_transform(im1).unsqueeze(0), image_transform(im2).unsqueeze(0)
    tens_1 = tens_1.to(device) - 0.5  # normalization as the input is expected to be in [-0.5, 0.5]

    tens_2 = tens_2.to(device) - 0.5  # normalization as the input is expected to be in [-0.5, 0.5]
    with torch.no_grad():
        _, _, z_1 = model(tens_1)
        _, _, z_2 = model(tens_2)

    deltas_z = [sub_z_2 - sub_z_1 for sub_z_1, sub_z_2 in zip(z_1, z_2)]
    with open(f"outputs/interpolation/info.txt", "w") as f:
        f.write(f"images paths: im1: {im1_path}\nim2: {im2_path}\n")
        f.write(f'alpha_steps: {alpha_steps}\nalpha_max: {alpha_max}\n')
    for i in range(alpha_steps):
        cur_z = [sub_z_1 + sub_delta_z * steps[i] for sub_z_1, sub_delta_z in zip(z_1, deltas_z)]
        cur_image = model.reverse(cur_z, reconstruct=True).cpu()
        utils.save_image(
            cur_image,
            f'outputs/interpolation/images/alpha_{round(steps[i].item(), 2)}.png',
            normalize=True,
            range=(-0.5, 0.5),
        )
    create_video_with_labels(base_path + '/video.mp4', [base_path + f'/images/alpha_{round(s.item(), 2)}.png' for s in
                                                                  steps], [f'alpha={round(s.item(), 2)}' for s in steps])


def create_video_with_labels(out_path, paths, labels, fps=2, codec='libx264', bitrate='16M'):
    video = imageio.get_writer(out_path, mode='I', fps=fps, codec=codec, bitrate=bitrate)
    images = [Image.open(path) for path in paths]
    font = ImageFont.truetype('f1.ttf', 16)
    for i in range(len(paths)):
        cur_draw = ImageDraw.Draw(images[i])
        cur_draw.text((70, 0), labels[i], font=font, fill=(255, 0, 0))
        video.append_data(np.asarray(images[i]))

    video.close()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    images_base = "outputs/interpolation"
    make_interpolation_video(args, images_base + "/im1.png", images_base + "/im2.png", model, device)


if __name__ == '__main__':
    main()

import math
import os
from glob import glob
from typing import List, Tuple
from PIL import Image, ImageFont, ImageDraw
import imageio
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from utils import get_args, compute_bpd
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


def get_logprob_stats(paths, model, device) -> Tuple[np.ndarray, np.ndarray]:
    logprobs_total, logprobs_last = [], []
    img_to_tensor = ToTensor()
    normal_dist = torch.distributions.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    for path in paths:
        img = Image.open(path)
        if img.width != 128 or img.height != 128:
            img = img.resize((128, 128))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = img_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            log_p, _, cur_z = model(tensor)
        logprobs_total.append(log_p)
        logprobs_last.append(torch.sum(normal_dist.log_prob(cur_z[-1]).view(1, -1), dim=1))
    return torch.cat(logprobs_total).cpu().numpy(), torch.cat(logprobs_last).cpu().numpy()


def plot_logprobs(roots, model, device, labels, save_dir, num_images=500):
    if not all(os.path.isfile(f'{save_dir}/{label}_{suf}_logprobs.npy') for label in labels for suf in ['total', 'last']):
        print(f"Computing logprobs...")
        types = ('*.png', '*.jpg', '*.jpeg')
        all_total_logprobs, all_last_logprobs = [], []
        for i, root in enumerate(roots):
            paths = []
            for type in types:
                paths += glob(root + '/' + type)
            paths = paths[:num_images]
            total_logprobs, last_logprobs = get_logprob_stats(paths, model, device)
            print("Finished " + labels[i], flush=True)
            np.save(f"{save_dir}/{labels[i]}_total_logprobs.npy", total_logprobs)
            np.save(f"{save_dir}/{labels[i]}_last_logprobs.npy", last_logprobs)
            all_total_logprobs.append(total_logprobs)
            all_last_logprobs.append(last_logprobs)
        rand_total, rand_last = get_rand_images_logprob(model, device, save_dir, num_images)
        all_total_logprobs.append(rand_total)
        all_last_logprobs.append(rand_last)
        print("Finished All successfully")
    else:
        print("Loading logprobs...")
        all_total_logprobs, all_last_logprobs = [], []
        for label in labels:
            cur_total = np.load(f"{save_dir}/{label}_total_logprobs.npy")
            cur_last = np.load(f"{save_dir}/{label}_last_logprobs.npy")
            all_total_logprobs.append(cur_total)
            all_last_logprobs.append(cur_last)
    print("Plotting")
    scale = 10 ** -5
    for prob, logprob_arr in [('total', all_total_logprobs), ('last', all_last_logprobs)]:
        plt.figure()
        logprob_arr = [arr * scale for arr in logprob_arr]
        min_bin = min([np.min(logprobs) for logprobs in logprob_arr])
        max_bin = max([np.max(logprobs) for logprobs in logprob_arr])
        print(prob)
        print("Min:", min_bin, "Max:", max_bin)
        bins = np.linspace(min_bin, max_bin, 200)
        data = {}
        for i in range(len(labels)):
            plt.hist(logprob_arr[i], bins=bins, label=labels[i], alpha=0.6, density=True)
            data[labels[i]] = {'min': str(np.min(logprob_arr[i])), 'max': str(np.max(logprob_arr[i])),
                               'mean': str(np.mean(logprob_arr[i])), 'std': str(np.std(logprob_arr[i])),
                               'median': str(np.median(logprob_arr[i])), 'var': str(np.var(logprob_arr[i]))}
        plt.legend(loc='upper right')
        plt.ylabel('Count [Normalized]')
        plt.xlabel(f'Logprob X {scale}')
        plt.title(f'{prob} layers Logprobs'.title())
        plt.savefig(f"{save_dir}/{prob}_logprobs.png")
        with open(f"{save_dir}/{prob}_logprobs.json", 'w') as f:
            data['scale'] = scale
            json.dump(data, f, indent=4)
        plt.close()


def get_rand_images_logprob(model, device, save_dir, num_images=500)-> Tuple[np.ndarray, np.ndarray]:
    images = torch.randn(num_images, 3, 128, 128, device=device)
    total_logprobs, last_logprobs = [], []
    normal_dist = torch.distributions.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

    for i in range(num_images):
        with torch.no_grad():
            log_p, _, cur_z = model(images[i].unsqueeze(0))
        total_logprobs.append(log_p)
        last_logprobs.append(torch.sum(normal_dist.log_prob(cur_z[-1]).view(1, -1), dim=1))
    np_total, np_last = torch.cat(total_logprobs).cpu().numpy(), torch.cat(last_logprobs).cpu().numpy()
    np.save(f"{save_dir}/random_total_logprobs.npy", np_total)
    np.save(f"{save_dir}/random_last_logprobs.npy", np_last)
    return np_total, np_last


def produce_partial_latents_images(input_im_path, args, model, device, save_dir='outputs/partial_latents', change_last=False):
    to_tens = Compose([Resize(args.img_size), ToTensor()])
    img = to_tens(Image.open(input_im_path)).to(device).unsqueeze(0) - 0.5
    with torch.no_grad():
        _, _, z_list = model(img)
    suffix = '_changed' if change_last else ''
    if change_last:
        z_list[-1] = torch.randn_like(z_list[-1])
    sample_from_model(z_list, model, f'{save_dir}/no_erase{suffix}.png', reconstruct=True)
    for i in range(len(z_list) - 1):
        z_list[i].zero_()
        sample_from_model(z_list, model, f'{save_dir}/erase_{i}{suffix}.png', reconstruct=True)


def grayscale_2_rgb(img_tensor):
    if img_tensor.shape[0] == 1:
        img_tensor = torch.cat([img_tensor, img_tensor, img_tensor], dim=0)

    return img_tensor


class PathsDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


class RandomDataset(Dataset):
    def __init__(self, img_size, num_images, transform=None, uniform=False):
        self.img_size = img_size
        self.num_images = num_images
        self.transform = transform
        self.uniform = uniform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.uniform:
            img = torch.rand(*self.img_size)
        else:
            img = torch.randn(*self.img_size)

        if self.transform:
            img = self.transform(img)
        return img, 0


def quantize_image(img, n_bits):
    """
    assuming the input is in [0, 1] of 8 bit images for each channel
    """
    if n_bits < 8:
        img = img * 255
        img = torch.floor(img / (2 ** (8 - n_bits)))
        img /= (2 ** n_bits)
    return img - 0.5


def get_bpd_of_images(args, model, device, paths=None, **kwargs):
    to_tensor = Compose([Resize((args.img_size, args.img_size)), ToTensor(), lambda tens: quantize_image(tens, args.n_bits)])
    if 'random' in kwargs:
        scale = 0.5 if not 'scale' in kwargs else kwargs['scale']
        n = kwargs['random']
        dset = RandomDataset((3, args.img_size, args.img_size), n, lambda tens: tens * scale)
    elif 'uniform' in kwargs:
        n = kwargs['uniform']
        dset = RandomDataset((3, args.img_size, args.img_size), n, transform=lambda tens: tens - 0.5, uniform=True)
    else:
        dset = PathsDataset(paths, transform=to_tensor)
        n = len(paths)
    dl = DataLoader(dset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return compute_bpd(2 ** args.n_bits, args.img_size, model, device, dl)


def get_bpd_ood(args, model, device, save_dir='outputs/nll'):
    ds_root = '/home/yandex/AMNLP2021/malnick/datasets'
    roots = [ds_root + '/ffhq_256_samples', ds_root + '/cars_train', ds_root + '/chest_xrays/images',
             ds_root + '/celebA/celeba/img_align_celeba']
    num_images = 512
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for i, root in enumerate(roots):
        cur_paths = []
        for type in types:
            cur_paths += glob(root + '/' + type)
        cur_paths = cur_paths[:num_images]
        paths.append(cur_paths)
    labels = ['ffhq', 'cars', 'chest_xrays', 'celebA']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = {}
    for i in range(len(paths)):
        cur_bpd = get_bpd_of_images(args, model, device, paths[i])
        print(f'BPD of {labels[i]}: {cur_bpd}')
        data[labels[i]] = cur_bpd

    for scale in [0.1 * i for i in range(1, 11)]:
        cur_bpd = get_bpd_of_images(args, model, device, random=num_images, scale=scale)
        print(f'BPD of random with scale {scale}: {cur_bpd}')
        data[f'random_{scale}'] = cur_bpd
    uni_bpd = get_bpd_of_images(args, model, device, uniform=num_images)
    print(f'BPD of uniform: {uni_bpd}')
    data['uniform'] = uni_bpd
    with open(f'{save_dir}/bpd_ood.json', 'w') as f:
        json.dump(data, f, indent=4)
    with open(f'{save_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    return data


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    data = get_bpd_ood(args, model, device, save_dir='outputs/bpd_5_bits')
    print(data)


if __name__ == '__main__':
    main()

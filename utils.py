import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Union

from matplotlib import pyplot as plt

from model import Glow
import torch
from easydict import EasyDict
import json
import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_args() -> EasyDict:
    parser = ArgumentParser(description="Glow trainer", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch", help="batch size", type=int)
    parser.add_argument("--iter", help="maximum iterations", type=int)
    parser.add_argument(
        "--n_flow", help="number of flows in each block", type=int
    )
    parser.add_argument("--n_block", help="number of blocks", type=int)
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
        default=None
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive", default=None
    )
    parser.add_argument("--n_bits", help="number of bits", type=int)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--img_size", help="image size", type=int)
    parser.add_argument("--temp", help="temperature of sampling", type=float)
    parser.add_argument("--n_sample", help="number of samples", type=int)
    parser.add_argument("--ckpt_path", help='Path to checkpoint for model')
    parser.add_argument("--opt_path", help='Path to checkpoint for optimizer')
    celeba_path = '/home/yandex/AMNLP2021/malnick/datasets/celebA'
    ffhq_path = '/home/yandex/AMNLP2021/malnick/datasets/ffhq-128'
    parser.add_argument("--path", metavar="PATH", help="Path to image directory")
    parser.add_argument('--eval', action='store_true', help='Use for evaluating a model', default=None)
    parser.add_argument('--sample_name', help='Name of sample size in case of evaluation')
    parser.add_argument('--exp_name', help='Name experiment for saving dirs')
    parser.add_argument('--num_workers', help='Number of worker threads for dataloader', type=int)
    parser.add_argument('--config', help='Name of json config file (optional) cmd will be overriden by file option')
    parser.add_argument('--devices', help='number of gpu devices to use', type=int)
    parser.add_argument('--forget_path', help='path to forget dataset root')
    parser.add_argument('--forget_size', help='Number of images to forget', type=int)
    parser.add_argument('--forget_every', help='learn a forget batch every <forget_every> batches', type=int)
    parser.add_argument('--save_every', help='number of steps between model and optimizer saving periods', type=int)
    parser.add_argument('--data_split', help='optional for data split, one of [train, val, test, all]')

    args = parser.parse_args()
    out_dict = EasyDict()
    if args.config:
        with open(args.config, "r") as in_j:
            config_dict = json.load(in_j)
        out_dict.update(config_dict)
    args_dict = vars(args)
    for k in args_dict:
        if args_dict[k] or k not in out_dict:
            out_dict[k] = args_dict[k]

    assert out_dict.path
    if 'ffhq' in out_dict.path.lower():
        out_dict.path = ffhq_path
    elif 'celeba' in out_dict.path.lower():
        out_dict.path = celeba_path

    return EasyDict(out_dict)


def make_exp_dir(exp_name, exist_ok=False) -> str:
    if not exp_name:
        exp_name = chr(random.randint(97, ord('z'))) + chr(random.randint(97, ord('z')))
    os.makedirs(f'experiments/{exp_name}/samples', exist_ok=exist_ok)
    os.makedirs(f'experiments/{exp_name}/checkpoints', exist_ok=exist_ok)
    return exp_name


def save_dict_as_json(save_dict, save_path):
    with open(save_path, 'w') as out_j:
        json.dump(save_dict, out_j, indent=4)


def get_dataset(data_root_path, image_size, **kwargs):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    if 'celeba' in data_root_path.lower():
        split = 'all' if not 'split' in kwargs else kwargs['split']
        ds = datasets.CelebA(data_root_path, split, transform=transform, download=False, target_type='identity')
    else:
        ds = datasets.ImageFolder(data_root_path, transform=transform)
    return ds


def get_dataloader(data_root_path, batch_size, image_size, num_workers=8, dataset=None) -> DataLoader:
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def sample_data(data_root_path, batch_size, image_size, num_workers=8, dataset=None, **kwargs):
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size, **kwargs)
    loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
            loader = iter(loader)
            yield next(loader)


def compute_bpd(n_bins, img_size, model, device, data_loader):
    """
    Computation of bits per dimension as done in Glow, meaning we:
    - compute the negative log likelihood of the data
    - add to the log likelihood the dequantization term -Mlog(a), where M=num_pixels, a = 1/n_bins
    - divide by log(2) for change base of the log used in the nll
    - divide by num_pixels
    :param n_bins: number of bins the data is quantized into.
    :param data_loader: data loader for the data. the data is expected to be qunatized to n_bins.
    :return: bits per dimension
    """
    nll = 0.0
    total_images = 0
    for batch in data_loader:
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            log_p, logdet, _ = model(x)
        nll -= torch.sum(log_p + logdet).item()
        total_images += x.shape[0]
    nll /= total_images
    M = img_size * img_size * 3
    bpd = (nll + (M * math.log(n_bins))) / (math.log(2) * M)
    return bpd


def load_model(args, device, training=False) -> Union[Glow, torch.nn.DataParallel]:
    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model = torch.nn.DataParallel(model_single)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage))
    model.to(device)
    if training:
        return model
    else:
        return model_single


def quantize_image(img, n_bits):
    """
    assuming the input is in [0, 1] of 8 bit images for each channel
    """
    if n_bits < 8:
        img = img * 255
        img = torch.floor(img / (2 ** (8 - n_bits)))
        img /= (2 ** n_bits)
    return img - 0.5


def json_2_bar_plot(json_path, out_path, **kwargs):
    with open(json_path, 'r') as in_j:
        data = json.load(in_j)

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(list(data.keys()), list(data.values()))

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')
    if 'title' in kwargs:
        # Add Plot Title
        ax.set_title(kwargs['title'], loc='left')
    plt.savefig(out_path)
    plt.close()

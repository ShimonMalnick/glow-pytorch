from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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
    celeba_path = '/home/yandex/AMNLP2021/malnick/datasets/celebA/celeba'
    ffhq_path = '/home/yandex/AMNLP2021/malnick/datasets/ffhq-128'
    parser.add_argument("--path", metavar="PATH", help="Path to image directory")
    parser.add_argument('--eval', action='store_true', help='Use for evaluating a model', default=None)
    parser.add_argument('--sample_name', help='Name of sample size in case of evaluation')
    parser.add_argument('--exp_name', help='Name experiment for saving dirs')
    parser.add_argument('--num_workers', help='Number of worker threads for dataloader', type=int)
    parser.add_argument('--config', help='Name of json config file (optional) cmd will be overriden by file option')
    parser.add_argument('--devices', help='number of gpu devices to use', type=int)

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


def get_dataset(data_root_path, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    return datasets.ImageFolder(data_root_path, transform=transform)


def get_dataloader(data_root_path, batch_size, image_size, num_workers=8, dataset=None) -> DataLoader:
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def sample_data(data_root_path, batch_size, image_size, num_workers=8):
    dataset = get_dataset(data_root_path, image_size)
    loader = get_dataloader(data_root_path, batch_size, image_size, num_workers)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
            loader = iter(loader)
            yield next(loader)
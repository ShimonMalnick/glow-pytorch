import math
import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from typing import Union, List
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize, Compose, Resize, ToTensor, RandomHorizontalFlip
from datasets import CelebAPartial
from model import Glow
import torch
from easydict import EasyDict
import json
import torchvision.datasets as vision_datasets
from torch.utils.data import DataLoader
from arcface_model import Backbone
from torchvision.models import resnet50

# Constants
CELEBA_ROOT = "/a/home/cc/students/cs/malnick/thesis/datasets/celebA"
FFHQ_ROOT = '/a/home/cc/students/cs/malnick/thesis/datasets/ffhq-128'
CELEBA_NUM_IDENTITIES = 10177
CELEBA_MALE_ATTR_IDX = 20
CELEBA_GLASSES_ATTR_IDX = 15


def get_args(**kwargs) -> EasyDict:
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
    parser.add_argument("--path", metavar="PATH", help="Path to image directory")
    parser.add_argument('--eval', action='store_true', help='Use for evaluating a model', default=None)
    parser.add_argument('--sample_name', help='Name of sample size in case of evaluation')
    parser.add_argument('--exp_name', help='Name experiment for saving dirs')
    parser.add_argument('--num_workers', help='Number of worker threads for dataloader', type=int)
    parser.add_argument('--config', '-c',
                        help='Name of json config file (optional) cmd will be overriden by file option')
    parser.add_argument('--devices', help='number of gpu devices to use', type=int)
    parser.add_argument('--log_every', help='run heavy logs every <log_every> iterations', type=int)
    if kwargs.get('forget', False):
        if kwargs.get('forget_attribute', False):
            parser.add_argument('--forget_attribute', help='which attribute to forget', choices=['male', 'glasses'])
        else:
            parser.add_argument('--forget_images', help='path to images to forget')
            parser.add_argument('--forget_identity', help='Identity to forget', type=int)
            parser.add_argument('--forget_size', help='Number of images to forget', type=int)

        parser.add_argument('--forget_every', help='learn a forget batch every <forget_every> batches', type=int)
        parser.add_argument('--save_every', help='number of steps between model and optimizer saving periods', type=int)
        parser.add_argument('--data_split', help='optional for data split, one of [train, val, test, all]')
        parser.add_argument('--forget_baseline', help='Whether running baseline forget experiment, meaning just'
                                                      ' forgetting with no regularization', type=bool)
        parser.add_argument('--forget_thresh', help='Threshold on forgetting procedure. when BPD of a batch of forget '
                                                    'images reaches more then base_bpd * forget_thresh, the finetuning '
                                                    'is stopped.', type=float)
        parser.add_argument('--num_ref_batches', help='number of reference batches from the remember data', type=int)
        parser.add_argument('--adaptive_loss', action='store_true', help='Whether to use adaptive loss weights for the '
                                                                         'forget examples', default=None)
        parser.add_argument('--forget_lr', help='Learning rate for the forget optimizer', type=float)
        parser.add_argument('--scheduler_step', help='Learning rate scheduler step size for the forget optimizer',
                            type=int)
        parser.add_argument('--scheduler_gamma', help='Learning rate scheduler gamma argument for StepLR scheduler',
                            type=float)
        parser.add_argument('--alpha', help='if given, training will be done with loss updates of both forget and'
                                            ' remember data in every update, by using'
                                            ' alpha * forget_loss + (1 - alpha) * remember_loss', type=float)

    args = parser.parse_args()
    out_dict = EasyDict()
    if args.config:
        with open(args.config, "r") as in_j:
            config_dict = json.load(in_j)
        out_dict.update(config_dict)
    args_dict = vars(args)
    for k in args_dict:
        if args_dict[k] is not None or k not in out_dict:
            out_dict[k] = args_dict[k]

    assert out_dict.get('forget_baseline', False) or out_dict.path
    if out_dict.path:
        if 'ffhq' in out_dict.path.lower():
            out_dict.path = FFHQ_ROOT
        elif 'celeba' in out_dict.path.lower():
            out_dict.path = CELEBA_ROOT

    return EasyDict(out_dict)


def save_dict_as_json(save_dict, save_path):
    with open(save_path, 'w') as out_j:
        json.dump(save_dict, out_j, indent=4)


def get_dataset(data_root_path, image_size, **kwargs):
    if 'transform' in kwargs:
        transform = kwargs['transform']
    else:
        transform = Compose(
            [
                Resize((image_size, image_size)),
                # CenterCrop(image_size),
                RandomHorizontalFlip(),
                ToTensor()])
    if 'celeba' in data_root_path.lower():
        assert 'data_split' in kwargs, 'data_split must be specified for celeba'
        if kwargs['data_split']:
            split = kwargs['data_split']
        ds = vision_datasets.CelebA(data_root_path, split, transform=transform, download=False, target_type='identity')
    else:
        ds = vision_datasets.ImageFolder(data_root_path, transform=transform)
    return ds


def get_dataloader(data_root_path, batch_size, image_size, num_workers=8, dataset=None, **kwargs) -> DataLoader:
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size, **kwargs)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def sample_data(data_root_path, batch_size, image_size, num_workers=8, dataset=None, **kwargs):
    if dataset is None:
        dataset = get_dataset(data_root_path, image_size, **kwargs)
    logging.info(f"loading Data Set of size: {len(dataset)}")
    loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = get_dataloader(data_root_path, batch_size, image_size, num_workers, dataset=dataset)
            loader = iter(loader)
            yield next(loader)


def compute_dataset_bpd(n_bins, img_size, model, device, dataset, reduce=True) -> Union[float, torch.Tensor]:
    """
    Computation of bits per dimension as done in Glow, meaning we:
    - compute the negative log likelihood of the data
    - add to the log likelihood the dequantization term -Mlog(a), where M=num_pixels, a = 1/n_bins
    - divide by log(2) for change base of the log used in the nll
    - divide by num_pixels
    :param reduce: if true, return a float representing the bpd of the whole data set. if False, return the
    individual scores, i.e. reduced_bpd = sum(unreduced_bpd) / batch_size
    :param n_bins: number of bins the data is quantized into.
    :param device: the device to run the computation on.
    :param model: The model to use for computing the nll.
    :param img_size: the images size (square of img_sizeXimg_size)
    :param dataset: data set for the data. the data is expected to be qunatized to n_bins.
    :return: bits per dimension
    """
    nll = 0.0
    total_images = len(dataset)
    for i in range(total_images):
        x, _ = dataset[i]
        x = x.to(device).unsqueeze(0)
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        if reduce:
            nll -= torch.sum(log_p + logdet).item()
        else:
            cur_nll = - (log_p + logdet)
            if i == 0:
                nll = cur_nll
            else:
                nll = torch.cat((nll, cur_nll))
    M = img_size * img_size * 3
    bpd = (nll + (M * math.log(n_bins))) / (math.log(2) * M)
    if reduce:
        nll /= total_images
    return bpd


def compute_dataloader_bpd(n_bins, img_size, model, device, data_loader: DataLoader, reduce=True) \
        -> Union[float, torch.Tensor]:
    """
    Computation of bits per dimension as done in Glow, meaning we:
    - compute the negative log likelihood of the data
    - add to the log likelihood the dequantization term -Mlog(a), where M=num_pixels, a = 1/n_bins
    - divide by log(2) for change base of the log used in the nll
    - divide by num_pixels
    :param device: the device to run the computation on.
    :param model: The model to use for computing the nll.
    :param img_size: the images size (square of img_sizeXimg_size)
    :param n_bins: number of bins the data is quantized into.
    :param data_loader: data loader for the data. the data is expected to be qunatized to n_bins.
    :return: bits per dimension
    """
    nll = 0.0
    total_images = 0
    for idx, batch in enumerate(data_loader, 1):
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            log_p, logdet, _ = model(x)
            logdet = logdet.mean()
        if reduce:
            nll -= torch.sum(log_p + logdet).item()
        else:
            cur_nll = - (log_p + logdet)
            if total_images == 0:
                nll = cur_nll.detach().cpu()
            else:
                nll = torch.cat((nll, cur_nll.detach().cpu()))
        total_images += x.shape[0]
        logging.debug(f"finished batch: {idx}/{len(data_loader)}")
    M = img_size * img_size * 3
    bpd = (nll + (M * math.log(n_bins))) / (math.log(2) * M)
    if reduce:
        nll /= total_images
    return bpd


def load_model(args, device, training=False) -> Union[Glow, torch.nn.DataParallel]:
    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model = torch.nn.DataParallel(model_single)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=lambda storage, loc: storage))
    model.to(device)
    logging.debug("Loaded model successfully")
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


def create_horizontal_bar_plot(data: Union[str, dict], out_path, **kwargs) -> None:
    """
    Given a data dictionary (or path to json containing the data), create a horizontal bar plot and save it
    in out_path
    :param data: data dictionary (or subclass of it), or path to json containing the data
    :param out_path: path to save the plot
    :param kwargs: 'title' can be passed as a string with the title
    :return: None
    """
    if isinstance(data, str):
        with open(data, 'r') as in_j:
            data = json.load(in_j)
    elif not isinstance(data, dict):
        raise ValueError('data must be a dictionary or a path to a json file')
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


def save_model_optimizer(args, iter_num, model, optimizer, save_prefix='experiments', last=False,
                         save_optim=True) -> None:
    if last:
        model_id = "last"
    else:
        model_id = str(iter_num + 1).zfill(6)
    torch.save(
        model.state_dict(), f'{save_prefix}/{args.exp_name}/checkpoints/model_{model_id}.pt'
    )
    if save_optim:
        torch.save(
            optimizer.state_dict(), f'{save_prefix}/{args.exp_name}/checkpoints/optim_{model_id}.pt'
        )


def gather_jsons(in_paths, keys_names, out_path, add_duplicate_names=False) -> None:
    d = {}
    for p, key_name in zip(in_paths, keys_names):
        with open(p, 'r') as in_j:
            data = json.load(in_j)
        if add_duplicate_names and key_name in d:
            key_name += '_addition'

        d[key_name] = data

    with open(out_path, 'w') as out_j:
        json.dump(d, out_j, indent=4)


def load_arcface(device=None) -> Backbone:
    model = Backbone(50, 0.6, 'ir_se').to('cpu')
    model.requires_grad_(False)
    cls_ckpt = "/a/home/cc/students/cs/malnick/thesis/glow-pytorch/models/arcface/model_ir_se50.pth"
    model.load_state_dict(torch.load(cls_ckpt, map_location='cpu'))
    if device:
        model = model.to(device)
    return model.eval()


def load_arcface_transform():
    return Compose([Resize((112, 112)),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5,
                               0.5))])


def compute_cosine_similarity(e1, e2, mean=False) -> torch.Tensor:
    """
    Compute cosine similarity matrix (or reduced mean) of two arcface embedding matrices
    :param e1: First embedding's matrix of shape (N_1, E)
    :param e2: Second embedding's matrix of shape (N_2, E)
    :param mean: if true, return a mean of the outputs matrix, i.e. a scalar
    :return: matrix E = e_1 @ e_2.T of shape (N_1, N_2)
    """
    e1 = e1
    e2 = e2
    e = e1 @ e2.T
    if mean:
        return e.mean()
    return e


def get_num_params(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in the given model
    :param model: torch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_resnet_for_binary_cls(pretrained=True, all_layers_trainable=True, device=None, num_outputs=1) -> resnet50:
    """
    Loads a pretrained resenet50 apart from the last classification layer (fc) that is changed to a new untrained layer
    with output size 1
    :param num_outputs: number of outputs of the new last layer
    :param pretrained: whether to load the pretrained weights or not
    :param device: device to load the model to
    :param all_layers_trainable: if True, all layers are trainable, otherwise only the last layer is trainable
    :return:
    """
    model = resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = all_layers_trainable
    model.fc = torch.nn.Linear(2048, num_outputs)  # 2048 is the number of activations before the fc layer
    if device:
        model = model.to(device)
    return model


def get_resnet_50_normalization():
    """
    This is the normalization according to:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    :return:
    """
    return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def get_partial_dataset(transform, target_type='identity', **kwargs) -> CelebAPartial:
    """
    Used for datasets when performing censoring/forgetting. this function is used both to obtain a dataset containing
    only some identities/images, or to obtain a dataset containing all iamges in celeba apart from certain
    identities/images. See documnetaion of CelebAPArtial for more details.
    """
    ds = CelebAPartial(root=CELEBA_ROOT, transform=transform, target_type=target_type, **kwargs)
    logging.info(f"len of dataset: {len(ds)}")
    return ds


def get_default_forget_transform(img_size, n_bits) -> Compose:
    transform = Compose([Resize((img_size, img_size)),
                         # RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, n_bits)])

    return transform


def images_to_gif(path: Union[str, List[str]], out_path, duration=300, **kwargs) -> None:
    """
    Given a path to a directory containing images, create a gif from the images using Pillow save function
    :param duration: duration in milliseconds for each image
    :param path: Either a str containing a path to a directory containing only images ,or list of paths to images
    :param out_path: path to save the gif
    :param kwargs: arguments to pass to the save function
    :return: None
    """
    if isinstance(path, str):
        path = glob(f"{path}/*")
    elif not isinstance(path, list):
        raise ValueError("path must be either a str or a list of str")
    images = [Image.open(p) for p in path]
    assert len(images) > 0, "No images found in directory"
    images[0].save(out_path, save_all=True, optimize=False, append_images=images[1:], loop=0,
                   duration=duration, **kwargs)

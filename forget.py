import math
import os
from model import Glow
import torch
from utils import get_args, load_model, quantize_image, make_exp_dir, save_dict_as_json, \
    save_model_optimizer, CELEBA_ROOT
from torch.utils.data import DataLoader
from typing import Iterator, Dict
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from train import calc_loss
from datasets import CelebAPartial, ForgetSampler
import wandb


def get_partial_dataset(transform, **kwargs) -> CelebAPartial:
    """
    Used for datasets when performing censoring/forgetting. this function is used both to obtain a dataset containing
    only some identities/images, or to obtain a dataset containing all iamges in celeba apart from certain
    identities/images. See documnetaion of CelebAPArtial for more details.
    """
    ds = CelebAPartial(root=CELEBA_ROOT, transform=transform, target_type="identity", **kwargs)
    print("len of dataset: ", len(ds))
    return ds


def get_data_iterator(ds: CelebAPartial, batch_size, num_workers=16) -> Iterator:
    sampler = None
    shuffle = True
    if len(ds) < batch_size:
        sampler = ForgetSampler(ds, batch_size)
        shuffle = None
    dl_iter = iter(DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))

    while True:
        try:
            yield next(dl_iter)

        except StopIteration:
            dl_iter = iter(
                DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, sampler=sampler))
            yield next(dl_iter)


def calc_batch_bpd(args, model, batch):
    """
    Calculates the batch-wise BPD of the model.
    :param args: arguments relevant to the model's parameters and input image size.
    :param model: model to calculate the BPD with.
    :param batch: batch to calculate the BPD of.
    :return: batch-wise BPD of the model.
    """
    n_bins = 2 ** args.n_bits
    M = args.img_size * args.img_size * 3
    with torch.no_grad():
        log_p, logdet, _ = model(batch)
    cur_nll = - torch.sum(log_p + logdet.mean()).item() / batch.shape[0]
    cur_bpd = (cur_nll + (M * math.log(n_bins))) / (math.log(2) * M)
    return cur_bpd


def forget(args, remember_iter: Iterator, forget_iter: Iterator, model: Glow, device, forget_optimizer,
           remember_optimizer):
    n_bins = 2 ** args.n_bits
    # remember_bpd_thresh = args.base_remember_bpd + 3 * args.base_remember_std
    for i in range(args.iter):
        if (i + 1) % args.forget_every == 0:
            loss_sign = -1.0
            loss_name = "forget"
            cur_batch, _ = next(forget_iter)
            optimizer = forget_optimizer
        else:
            loss_sign = 1.0
            loss_name = "remember"
            cur_batch, _ = next(remember_iter)
            optimizer = remember_optimizer
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss_sign * loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_bpd = calc_batch_bpd(args, model, cur_batch.detach().clone())

        wandb.log({f"{loss_name}": {"loss": loss.item(),
                                    "bpd": cur_bpd,
                                    "log_p": log_p.item(),
                                    "log_det": log_det.item(),
                                    "prob": log_p.item() + log_det.item()},
                   "batch_type": loss_sign
                   })

        if loss_name == "forget" and cur_bpd >= (args.base_forget_bpd * args.forget_thresh):
            print("breaking after {} iterations".format(i + 1))
            wandb.log({f"achieved_thresh": i + 1})
            break

        print(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; "
        )

        if (i + 1) % args.save_every == 0:
            save_forget_checkpoint(args, forget_optimizer, i, model, remember_optimizer)
    save_forget_checkpoint(args, forget_optimizer, 0, model, remember_optimizer, last=True)


def save_forget_checkpoint(args, forget_optimizer, iter_num, model, remember_optimizer, last=False):
    if last:
        model_id = "last"
    else:
        model_id = str(iter_num + 1).zfill(6)
    torch.save(
        model.state_dict(), f'experiments/forget/{args.exp_name}/checkpoints/model_{model_id}.pt'
    )
    torch.save(
        forget_optimizer.state_dict(),
        f'experiments/forget/{args.exp_name}/checkpoints/forget_optim_{model_id}.pt'
    )
    torch.save(
        remember_optimizer.state_dict(),
        f'experiments/forget/{args.exp_name}/checkpoints/regular_optim_{model_id}.pt'
    )


def args2dset_params(args, ds_type) -> Dict:
    """
    Returns a dictionary of parameters to be passed to the dataset constructor.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: dictionary contating the parameters to be passed to the dataset constructor
    """
    assert ds_type in ['forget', 'remember'], "ds_type must be one of 'forget' or 'remember'"
    out = {"include_only_identities": None,
           "exclude_identities": None,
           "include_only_images": None,
           "exclude_images": None}
    if 'data_split' in args and args.data_split in ['train', 'all']:
        out['split'] = args.data_split
    if ds_type == 'forget':
        if args.forget_identity:
            out["include_only_identities"] = [args.forget_identity]
        elif args.forget_images:
            assert args.forget_size, "must specify forget_size if forget_images is specified"
            out["include_only_images"] = os.listdir(args.forget_images)[:args.forget_size]
    if ds_type == 'remember':
        if args.forget_identity:
            out["exclude_identities"] = [args.forget_identity]
        elif args.forget_images:
            assert args.forget_size, "must specify forget_size if forget_images is specified"
            out["exclude_images"] = os.listdir(args.forget_images)[:args.forget_size]

    return out


def args2data_iter(args, ds_type, transform) -> Iterator:
    """
    Returns a data iterator for the dataset specified by ds_type.
    :param transform: transform to be applied to the dataset.
    :param args: arguments determining the images/identities to forget/remember.
    :param ds_type: one of 'forget' or 'remember'
    :return: data iterator for the dataset specified by ds_type
    """
    ds_params = args2dset_params(args, ds_type)
    ds = get_partial_dataset(transform=transform, **ds_params)
    return get_data_iterator(ds, args.batch, num_workers=args.num_workers)


def forget_baseline(forget_iter: Iterator, args, model, device, optimizer):
    n_bins = 2 ** args.n_bits
    for i in range(args.iter):
        cur_batch, _ = next(forget_iter)
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss * -1.0  # to forget instead of learning these examples

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_bpd = calc_batch_bpd(args, model, cur_batch.detach().clone())

        wandb.log({"loss": loss.item(),
                   "log_p": log_p.item(),
                   "log_det": log_det.item(),
                   "prob": log_p.item() + log_det.item(),
                   f"bpd": cur_bpd})

        if cur_bpd >= (args.base_forget_bpd * args.forget_thresh):
            print("breaking after {} iterations".format(i + 1))
            wandb.log({f"achieved_thresh": i + 1})
            break

        print(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f};")

        if (i + 1) % args.save_every == 0:
            save_model_optimizer(args, i, model, optimizer, save_prefix='experiments/forget')
    save_model_optimizer(args, 0, model, optimizer, save_prefix='experiments/forget', last=True)


def main():
    # os.environ["WANDB_DISABLED"] = "true"  # for debugging without wandb
    args = get_args(forget=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_name = make_exp_dir(args.exp_name, exist_ok=False, forget=True)
    print(args)
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    transform = Compose([Resize((args.img_size, args.img_size)),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         lambda img: quantize_image(img, args.n_bits)])
    forget_iter = args2data_iter(args, ds_type='forget', transform=transform)
    first_batch = next(forget_iter)[0]
    args.base_forget_bpd = calc_batch_bpd(args, model, first_batch.to(device))
    print("base forget bpd: ", args.base_forget_bpd)
    forget_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.forget_baseline:
        wandb.init(project="Dynamic Forget", entity="malnick", name=args.exp_name, config=args,
                   dir=f'experiments/forget/{args.exp_name}/wandb')
        save_dict_as_json(args, f'experiments/forget/{args.exp_name}/args.json')
        forget_baseline(forget_iter, args, model, device, forget_optimizer)
    else:
        remember_iter = args2data_iter(args, ds_type='remember', transform=transform)
        wandb.init(project="Dynamic Forget", entity="malnick", name=args.exp_name, config=args,
                   dir=f'experiments/forget/{args.exp_name}/wandb')
        save_dict_as_json(args, f'experiments/forget/{args.exp_name}/args.json')
        # args.base_remember_bpd = 0.609  # constant derived from the mean of BPD across all of celeba train
        # args.base_remember_std = 0.033  # constant derived from the std of BPD across all of celeba train
        remember_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        forget(args, remember_iter, forget_iter, model, device, forget_optimizer, remember_optimizer)


if __name__ == '__main__':
    main()

from model import Glow
import torch
from utils import get_args, load_model, quantize_image, sample_data, make_exp_dir, save_dict_as_json
from torch.utils.data import Subset
from typing import Iterator
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize
from train import calc_loss


def get_forget_dataloader(args) -> Iterator:
    transform = Compose([Resize((args.img_size, args.img_size)), RandomHorizontalFlip(), ToTensor(), lambda img : quantize_image(img, args.n_bits)])
    initial_ds = ImageFolder(args.forget_path, transform=transform)
    ds = Subset(initial_ds, [i % args.forget_size for i in range(max(args.batch, args.forget_size))])
    return sample_data(args.forget_path, args.batch, args.img_size, dataset=ds)


def forget(args, regular_iter: Iterator, forget_iter: Iterator, model: Glow, device, optimizer):
    n_bins = 2 ** args.n_bits
    forget_regularizer = 1.0
    for i in range(args.iter):
        loss_sign = 1.0
        if (i + 1) % args.forget_every == 0:
            print("Forget batch")
            loss_sign = -1.0 * forget_regularizer
            cur_batch, _ = next(forget_iter)
        else:
            print("regular batch")
            cur_batch, _ = next(regular_iter)
        cur_batch = cur_batch.to(device)
        log_p, logdet, _ = model(cur_batch + torch.rand_like(cur_batch) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        loss = loss_sign * loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Iter: {i + 1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; "
        )

        if (i + 1) % args.save_every == 0:
            torch.save(
                model.state_dict(), f'experiments/{args.exp_name}/checkpoints/model_{str(i + 1).zfill(6)}.pt'
            )
            torch.save(
                optimizer.state_dict(), f'experiments/{args.exp_name}/checkpoints/optim_{str(i + 1).zfill(6)}.pt'
            )


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_name = make_exp_dir(args.exp_name)
    print(args)
    save_dict_as_json(args, f'experiments/{args.exp_name}/args.json')
    model: torch.nn.DataParallel = load_model(args, device, training=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # if args.opt_path:
    #     optimizer.load_state_dict(torch.load(args.opt_path, map_location=lambda storage, loc: storage))
    regular_iter: Iterator = iter(sample_data(args.path, args.batch, args.img_size))
    forget_iter: Iterator = get_forget_dataloader(args)
    forget(args, regular_iter, forget_iter, model, device, optimizer)


if __name__ == '__main__':
    main()
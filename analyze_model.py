from typing import List

import torch
from utils import get_args
from torchvision import utils
from model import Glow
from train import calc_z_shapes


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


def sample_from_model(z_samples, model, out_name='samples.png'):
    with torch.no_grad():
        cur_tensors = model.reverse(z_samples).cpu()
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


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_different_temp(args, device)


if __name__ == '__main__':
    main()

from model import Glow
import torch


if __name__ == '__main__':
    # just as they did in the paper for celeba-Hq 256x256 Except for lu decomposition which i'm not sure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glow = Glow(3, 32, 6, affine=False, conv_lu=True).to(device)
    params = [param.nelement() for param in glow.parameters()]
    print("Params numbers: ", params)
    print("Total params: ", sum(params))

from model import gaussian_log_p_fixed, Glow
import torch
from types import MethodType


def fixed_forward(self, input):
    b_size, n_channel, height, width = input.shape
    squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
    out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
    logdet = 0
    for flow in self.flows:
        out, det = flow(out)
        logdet = logdet + det

    if self.split:
        out, z_new = out.chunk(2, 1)
        mean, log_sd = self.prior(out).chunk(2, 1)
        log_p = gaussian_log_p_fixed(z_new, mean, log_sd)
        log_p = log_p.view(b_size, -1).sum(1)

    else:
        zero = torch.zeros_like(out)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        log_p = gaussian_log_p_fixed(out, mean, log_sd)
        log_p = log_p.view(b_size, -1).sum(1)
        z_new = out

    return out, logdet, log_p, z_new


def get_fixed_model(model: Glow) -> Glow:
    for i in range(len(model.blocks)):
        model.blocks[i].forward = MethodType(fixed_forward, model.blocks[i])
    return model

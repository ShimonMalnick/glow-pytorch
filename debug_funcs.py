from model import Glow


def model_debugging(model: Glow, input):
    log_p_sum = 0
    logdet = 0
    out = input
    z_outs = []
    i = 1
    for block in model.blocks:
        if i == 1:
            total_logdet = 0
            b_size, n_channel, height, width = input.shape
            squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

            for j, flow in enumerate(block.flows):
                # if j == 1:
                #     break
                # print(flow.scale)
                cur_logdet = 0
                out, logdet = flow.actnorm(out)
                # print(f"logdet actnorm flow {j}: {logdet}")
                out, det1 = flow.invconv(out)
                # print(f"logdet invconv flow {j}: {det1}")
                out, det2 = flow.coupling(out)
                # print(f"logdet coupling flow {j}: {det2}")
                cur_logdet += logdet + det1
                if det2 is not None:
                    cur_logdet += det2
                print(f"cur logdet flow {j}: {cur_logdet}")
                print("*" * 30)
            break
        out, det, log_p, z_new = block(out)
        print(f"block {i}")
        print(f"log_p: {log_p[:10]}")
        print(f"det: {det}")
        print("*" * 30)
        i += 1
        z_outs.append(z_new)
        logdet = logdet + det

        if log_p is not None:
            log_p_sum = log_p_sum + log_p

    return log_p_sum, logdet, z_outs
import time
import numpy as np
import torchprof
import torch
from thop import profile, clever_format


def compute_speed(model, input_size, device, iteration):
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    torch.cuda.synchronize()
    for _ in range(50):
        model(input)
        torch.cuda.synchronize()

    time_spent = []
    for _ in range(iteration):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            model(input)
        torch.cuda.synchronize()
        time_spent.append(time.perf_counter() - t_start)
    torch.cuda.synchronize()
    elapsed_time = np.sum(time_spent)
    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        model(input)
    print(prof.display(show_events=False))

    print(prof.display(show_events=True))
    print('Elapsed time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Batch Speed Time: %.2f ms / iter    FPS: %.2f' % (elapsed_time / iteration * 1000, iteration / elapsed_time))
    print('Image Speed Time: %.2f ms / iter    FPS: %.2f' % (elapsed_time / iteration * 1000 / input_size[0],
                                                             iteration / elapsed_time * input_size[0]))


def eval_baseline():
    from model.baseline import BaselineNet
    model = BaselineNet(mode='large', width_mult=1.0, atrous3=False, atrous4=False, zoom_factor=8, classes=19)
    model.eval()
    inputs = torch.randn(1, 3, 713, 713)
    flops, params = profile(model, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.4f")
    print('Baseline Flops: {}, Params: {}'.format(flops, params))
    device = list(range(torch.cuda.device_count()))[0]
    print(device)
    input_size = tuple([8, 3, 713, 713])
    compute_speed(model, input_size, device, iteration=50)


def eval_mlms():
    from model.mlmsnet import MLMSNet
    model = MLMSNet(mode='large', width_mult=1.0, large=False, zoom_factor=8, use_msf=True, use_mlf=True)
    model.eval()
    inputs = torch.randn(1, 3, 713, 713)
    flops, params = profile(model, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.4f")
    print('MLMS Flops: {}, Params: {}'.format(flops, params))
    device = list(range(torch.cuda.device_count()))[0]
    print(device)
    input_size = tuple([8, 3, 713, 713])
    compute_speed(model, input_size, device, iteration=50)


if __name__ == '__main__':
    eval_baseline()
    eval_mlms()

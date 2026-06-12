import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    valid = (valid >= 0.5)
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1px': (epe < 1).float().mean().item(),
        'train_3px': (epe < 3).float().mean().item()
    }
    return flow_loss, metrics


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

_window_cache = {}

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    device = img1.device
    
    cache_key = (window_size, channel, device)
    
    if cache_key not in _window_cache:
        window = create_window(window_size, channel)
        window = window.to(device).type_as(img1)
        _window_cache[cache_key] = window
    else:
        window = _window_cache[cache_key]

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# 💡 [핵심 수정] 마스크 영역 내의 진짜 PSNR만 계산하도록 수정
def psnr(img1, img2, mask=None):
    if mask is not None:
        # 마스크가 있는 경우: (B, C, H, W) 차원에서 픽셀 오차 합산 / 유효 픽셀 수
        mse = (((img1 - img2) * mask) ** 2).sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) * img1.shape[1] + 1e-8)
        mse = mse.view(-1, 1)
    else:
        # 마스크가 없는 경우: 기존 전체 해상도 평균
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
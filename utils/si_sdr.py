import torch
def single_si_sdr(estimate, target, norm=True, take_log=True):
    """ Compute SI-SDR for Single Speaker Seperation

    Args:
        estimate: Tensor
            shape = batch x time
        target: Tensor
            shape = batch x time
        norm: bool
            zero-mean norm or not, default True
        take_log: bool
    Returns:
        losses: Double
    """
    assert estimate.size() == target.size()
    assert target.ndim == 2
    EPS = 1e-8
    if norm:
        mean_estimate = torch.mean(estimate, dim=1, keepdim=True)
        mean_target = torch.mean(target, dim=1, keepdim=True)
        estimate = estimate - mean_estimate
        target = target - mean_target
    # shape = batch x 1 x time
    dot = torch.sum(estimate * target, dim=1, keepdim=True)
    # shape = batch x 1 x time
    s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
    scaled_target = dot * target / s_target_energy

    e_noise = estimate - scaled_target
    losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)

    if take_log:
        losses = 10 * torch.log10(losses + EPS)

    losses = losses.mean() 
    return losses


def multi_si_sdr(estimate, target, norm=True, take_log=True):
    """ Compute SI-SDR for Multiple Speaker Seperation

    Args:
        estimate: Tensor
            shape = batch x n_src x time
        target: Tensor
            shape = batch x n_src x time
        norm: bool
            zero-mean norm or not, default True
        take_log: bool
    Returns:
        losses: Double
    """
    EPS = 1e-8
    assert estimate.size() == target.size()
    assert target.ndim == 3
    if norm:
        mean_estimate = torch.mean(estimate, dim=2, keepdim=True)
        mean_target = torch.mean(target, dim=2, keepdim=True)
        estimate = estimate - mean_estimate
        target = target - mean_target
    # shape = batch x n_src x time
    pair_wise_dot = torch.sum(estimate * target, dim=2, keepdim=True)
    # shape = batch x n_src x time
    s_target_energy = torch.sum(target ** 2, dim=2, keepdim=True)
    scaled_target = pair_wise_dot * target / s_target_energy

    e_noise = estimate - scaled_target
    losses = torch.sum(scaled_target ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)

    if take_log:
        losses = 10 * torch.log10(losses + EPS)
    losses = torch.mean(losses, dim=-1)
    return losses



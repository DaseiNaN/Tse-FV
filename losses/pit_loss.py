import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PITNetLoss(object):
    def __init__(self, mix_mag, mix_phase, masks, s1_mag, s1_phase, s2_mag, s2_phase, device):
        self.device = device
        self.mix_mag  = mix_mag.to(self.device)
        self.mix_phase = mix_phase.to(self.device)
        self.masks = masks
        self.target_mag = [s1_mag.to(self.device), s2_mag.to(self.device)]
        self.target_phase = [s1_phase.to(self.device), s2_phase.to(self.device)]
        self.batch_size = mix_mag.shape[0]
        self.input_size = torch.Tensor([mix_mag.shape[1]] * self.batch_size).to(self.device)

    def loss(self, permute):
        # print(permute)
        loss_for_permute = []
        for s, t in enumerate(permute):
            refer_mag = self.target_mag[t] * F.relu(torch.cos(self.mix_phase - self.target_phase[t]))
            utt_loss = torch.sum(
                torch.sum(
                    torch.pow(self.masks[s] * self.mix_mag - refer_mag, 2), -1
                ), -1
            )
            loss_for_permute.append(utt_loss)
        loss_perutt = sum(loss_for_permute) / self.input_size
        return loss_perutt

    def compute(self):
        pscore = torch.stack(
            [self.loss(item) for item in permutations(range(2))]
        )
        min_perutt, _ = torch.min(pscore, dim=0)
        return torch.sum(min_perutt) / (2 * self.batch_size)



if __name__ == '__main__':
    x = torch.Tensor([1] * 3)
    y = torch.Tensor([2, 2, 2])
    temp = torch.stack([x, y])

    _1, _2 = torch.min(temp, dim=0)
    print(_1)
    for permute in permutations(range(2)):
        for s, t in enumerate(permute):
            print(s, t)

        
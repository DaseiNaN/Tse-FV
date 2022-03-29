import sys
sys.path.append('../')
import torch



class Loss(object):
    def __init__(self, mix_mag, target_mask, non_silent):
        super(Loss).__init__()
        self.mix_mag = mix_mag
        self.target_mask = target_mask
        self.non_silent = non_silent
        self.device = torch.device('cuda:0')

    def loss(self):
        '''
           mix_mag: B x TF x D
           target_mask: B x T x F
           non_silent: B x T x F 
        '''
        B, T, F = self.non_silent.shape
        
        # B x TF x spks
        target_embs = torch.zeros([B, T*F, 2],device=self.device)
        
        target_embs.scatter_(2, self.target_mask.view(B, T*F, 1), 1)
        
        # B x TF x 1
        self.non_silent = self.non_silent.view(B, T*F, 1)
        self.mix_mag = self.mix_mag * self.non_silent
        self.target_mask = target_embs * self.non_silent

        

        vt_v = torch.norm(torch.bmm(torch.transpose(self.mix_mag,1,2), self.mix_mag), p=2)**2
        vt_y = torch.norm(torch.bmm(torch.transpose(self.mix_mag,1,2),
                                    self.target_mask), p=2)**2
        yt_t = torch.norm(torch.bmm(torch.transpose(self.target_mask,1,2),
                                    self.target_mask), p=2)**2
        
        return (vt_v-2*vt_y+yt_t)/torch.sum(self.non_silent)
import torch
import torch.nn.functional as F 

class L2DistanceLoss(torch.nn.Module):
    def __init__(self):
        super(L2DistanceLoss, self).__init__()

    def forward(self, zis1, zis2):
        l2_distance = torch.norm(zis1 - zis2, p=2, dim=1)
        return l2_distance.mean()
    
    

# Copyright (C) 2023 Elif Cansu YILDIZ
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn.functional as F
from medai.utils.transforms import HorizontalFlip

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, euclidian_distance, label):
        
        loss_contrastive = torch.mean( (1-label) * torch.pow(euclidian_distance, 2 ) + 
                                       (label) * torch.clamp(self.margin - torch.pow(euclidian_distance, 2), min=0.0) )
        
        return loss_contrastive

class ContrastiveLoss_(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss_, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        
        euclidian_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        loss_contrastive = torch.mean( (1-label) * torch.pow(euclidian_distance, 2 ) + 
                                       (label) * torch.clamp(self.margin - torch.pow(euclidian_distance, 2), min=0.0) )
        
        return loss_contrastive
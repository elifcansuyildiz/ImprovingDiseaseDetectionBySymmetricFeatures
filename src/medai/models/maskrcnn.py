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

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import Tensor
from torchvision import models
import torchvision
from types import MethodType
from torchinfo import summary
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from medai.config import logger

class MaskRCNN(nn.Module):
    def __init__(self, 
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5):
        super(MaskRCNN, self).__init__()
        # load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                                        box_score_thresh=box_score_thresh, 
                                                                        box_nms_thresh=box_nms_thresh)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        num_classes = 2
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                hidden_layer,
                                                                num_classes)
        
    def forward(self, images, targets):
        #images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        return loss_dict
    
    def predict(self, images):
        #images = list(image for image in images)
        outputs = self.model(images)
        return outputs
    
    
class Classifier(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, logging_dataset=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logging_dataset = logging_dataset
        self.val_pred_labels, self.val_target_labels = torch.asarray([]), torch.asarray([])
        self.test_pred_labels, self.test_target_labels = torch.asarray([]), torch.asarray([])
        self.logged_images = {}
        #self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        loss_dict = self.model(images, targets)
        loss = sum(loss_val for loss_val in loss_dict.values())
                
        #print(f"pred_label: {pred_label} target_label: {target_label}")
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss/train", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        #outputs = self.model.predict(images)
    
    def configure_optimizers(self):
        if self.scheduler is not None and self.scheduler != "None":
            logger.info(f"scheduler: {self.scheduler} \noptimizer: {self.optimizer}")
            return {
                    "optimizer": self.optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.1),
                        "monitor": "loss/train",
                        "frequency": 1,
                        "interval": "epoch",
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
            }
        logger.info(f"scheduler: {self.scheduler} \noptimizer: {self.optimizer}")
        return self.optimizer

    
if __name__ == "__main__":
    pass
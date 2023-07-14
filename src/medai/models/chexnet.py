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

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        self.densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        print(self.densenet.classifier)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = torch.nn.Linear(in_features=in_features*7*7, out_features=1)
        self.flatten_layer = nn.Flatten()
        
    def forward(self, image):
        conv_out = self.densenet.features(image)
        conv_out = F.relu(conv_out, inplace=True)
        out = self.densenet.classifier(self.flatten_layer(conv_out))

        return out
    
    def forward_and_compute_heatmap(self, image):
        with torch.no_grad():
            conv_out = self.densenet.features(image)
            conv_out = F.relu(conv_out, inplace=True)
            #print(conv_out.shape)
            classification_out = self.densenet.classifier(self.flatten_layer(conv_out))
            # Class Activation Map (CAM)
            #print(classification_out.shape)
            weights = self.densenet.classifier.weight.reshape(-1, 7, 7)
            # CAM (Class Activation Map)
            heatmap = torch.einsum("bcwh,cwh->bwh", conv_out, weights)
            #print(image.shape)
            heatmap = F.interpolate(heatmap.unsqueeze(dim=1), size=(image.shape[-2], image.shape[-1]), mode="bicubic")
            #print("heatmap.shape", heatmap.shape)
            #heatmap = heatmap.reshape(-1, image.shape[-2], image.shape[-1])
            return classification_out, heatmap
    
class Classifier(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, logging_dataset=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = BCEWithLogitsLoss(pos_weight=torch.tensor([437/1805]))  ## HARDCODED ratio: healthy(negative=0)/disease(positive=1)

        self.logging_dataset = logging_dataset
        self.val_pred_labels, self.val_target_labels = torch.asarray([]), torch.asarray([])
        self.test_pred_labels, self.test_target_labels = torch.asarray([]), torch.asarray([])
        self.logged_images = {}
        #self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        pred_label = self.model(img)
        with torch.no_grad():
            #target_label = torch.max(target_map.reshape(len(target_map), -1), dim=1)[0].reshape(-1, 1)
            target_label = torch.where(torch.max(target_map.reshape(len(target_map), -1), dim=1)[0] > 0, 1.0, 0.0).reshape(-1, 1)
        
        #print(f"pred_label: {pred_label} target_label: {target_label}")

        loss = self.loss_function(pred_label, target_label)  
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        pred_label, pred_heatmap = self.model.forward_and_compute_heatmap(img)

        self.batch_size = len(img)
            
        with torch.no_grad():
            target_label = torch.where(torch.max(target_map.reshape(len(target_map), -1), dim=1)[0] > 0, 1.0, 0.0).reshape(-1, 1)
        #print(f"pred_label: {pred_label} target_label: {target_label}")

        loss = self.loss_function(pred_label, target_label)
        
        self.log("loss/validation", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.val_pred_labels = torch.hstack((self.val_pred_labels, torch.sigmoid(pred_label.detach().cpu())))
            self.val_target_labels = torch.hstack((self.val_target_labels, target_label.detach().cpu()))
            self.logged_images = {"target_probmap":target_map.detach().cpu(),
                                  "pred_probmap": pred_heatmap.detach().cpu()}
        return loss
    
    def test_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        pred_label, pred_heatmap = self.model.forward_and_compute_heatmap(img)
        
        self.batch_size = len(img)
        
        with torch.no_grad():
            target_label = torch.where(torch.max(target_map.reshape(len(target_map), -1), dim=1)[0] > 0, 1.0, 0.0).reshape(-1, 1)

        loss = self.loss_function(pred_label, target_label)
        
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.test_pred_labels = torch.hstack((self.test_pred_labels, torch.sigmoid(pred_label.detach().cpu()).flatten()))
            self.test_target_labels = torch.hstack((self.test_target_labels, (target_label.detach().cpu()).flatten()))
            self.logged_images = {"target_probmap":target_map.detach().cpu(),
                                  "pred_probmap": pred_heatmap.detach().cpu()}
        return loss

    def configure_optimizers(self):
        return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.1),
                    "monitor": "loss/validation",
                    "frequency": 1,
                    "interval": "epoch",
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
        }

    
if __name__ == "__main__":
       
    x = torch.rand(4, 3, 224, 224)
    model = DenseNet()
    output, heatmap = model.forward_and_compute_heatmap(x)
    print(f"output.shape: {output.shape}")
    print(output)
    print(model.densenet)
    
    contrastive_loss_function = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    classifier = Classifier(model, 
                            optimizer,
                            scheduler="ReduceLROnPlateau")
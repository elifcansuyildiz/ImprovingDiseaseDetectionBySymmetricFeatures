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
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import torchvision
from types import MethodType
from torchinfo import summary

class Classifier2(pl.LightningModule):
    """Classifier for training and evaluating the SASN_vanilla model without contrastive learning part (Siamese Comparison Module)
    """
    def __init__(self, model, optimizer, scheduler=None, logging_dataset=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = BCEWithLogitsLoss() 
        self.scheduler = scheduler
        self.logging_dataset = logging_dataset
        self.val_pred_labels, self.val_target_labels = torch.asarray([]), torch.asarray([])
        self.test_pred_labels, self.test_target_labels = torch.asarray([]), torch.asarray([])
        self.logged_images = {}
        self.process_half_image = False
        #self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        probability_map =  self.model(img, flipped_img)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-1], probability_map.shape[-1]), mode="area")

        loss = self.loss_function(probability_map, target_probmap)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss/train", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        probability_map = self.model(img, flipped_img)
        
        #print(f"probability_map.shape: {probability_map.shape} \ndistance_map.shape: {distance_map.shape}")
        #print(f"probability_map: {probability_map} \ndistance_map: {distance_map}")

        self.batch_size = len(img)

        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-1], probability_map.shape[-1]), mode="area")

        loss = self.loss_function(probability_map, target_probmap)
        
        self.log("loss/validation", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.val_pred_labels = torch.hstack((self.val_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.val_target_labels = torch.hstack((self.val_target_labels, torch.where(torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0] > 0.5, 1.0, 0.0)))
            self.logged_images = {"target_probmap":target_map.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu())}
        return loss
    
    def test_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        probability_map = self.model(img, flipped_img)
        
        self.batch_size = len(img)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-1], probability_map.shape[-1]), mode="area")
        
        loss = self.loss_function(probability_map, target_probmap)
        
        self.log("loss/test", loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.test_pred_labels = torch.hstack((self.test_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.test_target_labels = torch.hstack((self.test_target_labels, torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0]))
            self.logged_images = {"target_probmap":target_probmap.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu())}
        return loss

    def configure_optimizers(self):
        if self.scheduler is not None and self.scheduler != "None":
            print("-------------> lr scheduler: ", self.scheduler)
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
        print("-------------> lr scheduler: ", self.scheduler)
        return self.optimizer
    
class Classifier3(pl.LightningModule):
    """Classifier for training and evaluating the SASN_split model without contrastive learning part (Siamese Comparison Module)
    """
    def __init__(self, model, optimizer, scheduler=None, logging_dataset=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = BCEWithLogitsLoss()
        self.scheduler = scheduler
        self.logging_dataset = logging_dataset
        self.val_pred_labels, self.val_target_labels = torch.asarray([]), torch.asarray([])
        self.test_pred_labels, self.test_target_labels = torch.asarray([]), torch.asarray([])
        self.logged_images = {}
        self.process_half_image = True
        #self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        
        probability_map = self.model(image_left, image_right_flipped)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")

        loss = self.loss_function(probability_map, target_probmap)   
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss/train", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        
        probability_map = self.model(image_left, image_right_flipped)

        self.batch_size = len(img)

        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")

        loss = self.loss_function(probability_map, target_probmap)
        
        self.log("loss/validation", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.val_pred_labels = torch.hstack((self.val_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.val_target_labels = torch.hstack((self.val_target_labels, torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0]))
            self.logged_images = {"target_probmap":target_map.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu())}
        return loss
    
    def test_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        
        probability_map = self.model(image_left, image_right_flipped)
        
        self.batch_size = len(img)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")
        
        loss = self.loss_function(probability_map, target_probmap)
        
        self.log("loss/test", loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.test_pred_labels = torch.hstack((self.test_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.test_target_labels = torch.hstack((self.test_target_labels, torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0]))
            self.logged_images = {"target_probmap":target_probmap.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu())}
        return loss

    def configure_optimizers(self):
        if self.scheduler is not None and self.scheduler != "None":
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
        return self.optimizer
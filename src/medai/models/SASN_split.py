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
from torchvision.transforms.functional import hflip
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import torchvision
from types import MethodType
from torchinfo import summary

from medai.utils.loss import ContrastiveLoss
#from medai.utils.transforms import HorizontalFlip

class SiameseEncoding(nn.Module):
    def __init__(self):
        super(SiameseEncoding, self).__init__()
        
        self.densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        #self.latent_dim = self.densenet.classifier.in_features
        self.densenet.features.transition3 = torch.nn.Identity()
        self.densenet.features.denseblock4 = torch.nn.Identity()
        self.densenet.classifier = torch.nn.Identity()
        self.densenet.features.norm5 = torch.nn.Identity()
        self.densenet.forward = MethodType(densenet_forward, self.densenet)
        #print(self.densenet.features)
        #summary(self.densenet.cpu(), (4, 3, 512, 512), depth=10, mode="train")
        #for name, param in self.densenet.named_parameters():
        #    print(name, param.size())
        #print(self.densenet.features.denseblock3.denselayer24.conv2)
    
    def forward(self, img):
        img_encodings = self.densenet(img)
        return img_encodings

def densenet_forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    #out = F.adaptive_avg_pool2d(out, (1, 1))
    #out = torch.flatten(out, 1)
    #out = self.classifier(out)
    return out   

class SiameseFeatureFusion(nn.Module):
    def __init__(self):
        super(SiameseFeatureFusion, self).__init__()
        # Feature extractor model
        self.densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        #print(self.densenet.features.denseblock3.denselayer24.out)
        
        num_input_features = 1024
        num_output_features = 256
        self.batch_norm = nn.BatchNorm2d(num_input_features) #torch.nn.BatchNorm2d(num_features=0)
        self.relu = nn.ReLU(inplace=True)
        #self.conv = nn.Conv2d(2*num_input_features, num_output_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = nn.Conv2d(2*num_input_features, num_output_features, kernel_size=7, stride=1, padding=3, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # block 4
        self.denseblock4 = self.densenet.features.denseblock3
        
        self.last_conv = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        #print("--------> ", self.denseblock4.denselayer1.conv1.in_channels)
        #for name, param in self.denseblock4.named_parameters():
        #    print(name, param.size())
        
    def forward(self, img_encoding, img_f_encoding):
        ##print("SiameseFeatureFusion module \nimage_encoding.shape: ", img_encoding.shape)
        img_encoding, img_f_encoding = self.batch_norm(img_encoding), self.batch_norm(img_f_encoding)
        img_encoding, img_f_encoding = self.relu(img_encoding), self.relu(img_f_encoding)
        ##print("img_encoding.shape: ", img_encoding.shape)
        #Concatenate
        fused_encoding = torch.cat((img_encoding, img_f_encoding), dim=1)
        ##print("after cat fused_encoding.shape: ", fused_encoding.shape)
        fused_encoding = self.conv(fused_encoding)
        fused_encoding = self.avg_pool(fused_encoding)
        ##print("fused_encoding.shape: ", fused_encoding.shape)
        out = self.denseblock4(fused_encoding)
        ##print("denseblock4 out.shape: ", out.shape)
        
        #probability_map = self.sigmoid(self.last_conv(out))
        probability_map = self.last_conv(out)
        ##print("probability_map.shape: ", probability_map.shape)
        return probability_map

class SiameseFeatureComparison(nn.Module):
    def __init__(self):
        super(SiameseFeatureComparison, self).__init__()
        self.conv = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, bias=False)
        self.nonlinear_projection = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, bias=False), 
                                                  nn.BatchNorm2d(num_features=1),
                                                  nn.LeakyReLU(inplace=True))
        
    def forward(self, img_encoding, img_f_encoding):
        img_encoding, img_f_encoding = self.conv(img_encoding), self.conv(img_f_encoding)
        ##print("\nSiameseFeatureComparison module: \nimg_encoding.shape", img_encoding.shape)
        
        img_encoding, img_f_encoding = self.nonlinear_projection(img_encoding), self.nonlinear_projection(img_f_encoding)
        ##print("After projection img_encoding.shape", img_encoding.shape)
        # L2 Distance (Euclidian distance)
        batch_size, img_dim_r, img_dim_c = img_encoding.shape[0], img_encoding.shape[2], img_encoding.shape[3]
        distance_map = F.pairwise_distance(img_encoding.reshape(-1, 1), img_f_encoding.reshape(-1, 1)).reshape(batch_size, img_dim_r, img_dim_c)
        ##print("distance_map.shape: ", distance_map.shape)
        return distance_map

class SiameseNetwork(pl.LightningModule):
    
    def __init__(self, finetuning=True, include_feature_comparison=True):
        #super(SiameseNetwork, self).__init__()
        super().__init__()
        self.finetuning = finetuning
        self.include_feature_comparison = include_feature_comparison

        self.encoding = SiameseEncoding()
        self.feature_fusion = SiameseFeatureFusion()
        self.feature_comparison = SiameseFeatureComparison()

    def get_trainable_parameters(self):
        if self.finetuning:
            return self.parameters()
        else:
            return self.densenet.classifier.parameters()

    def forward(self, img, img_flipped):
              
        img_encodings = self.encoding(img)
        img_f_encodings = self.encoding(img_flipped)
        
        p1 = self.feature_fusion(img_encodings, img_f_encodings)
        p2 = self.feature_fusion(img_f_encodings, img_encodings)
        probability_map = torch.cat((p1, hflip(p2)), dim=3) # batch, channel, row, col
        
        if self.include_feature_comparison:
            dm = self.feature_comparison(img_encodings, img_f_encodings)
            distance_map = torch.cat((dm, hflip(dm)), dim=2) # batch, row, col
            return probability_map, distance_map
        else:
            return probability_map

class Classifier(pl.LightningModule):
    def __init__(self, model, optimizer, contrastive_loss, lmbda=0.5, scheduler=None, logging_dataset=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = BCEWithLogitsLoss()
        self.contrastive_loss_function = contrastive_loss #ContrastiveLoss2(margin=margin)
        self.lmbda = lmbda
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
        
        probability_map, distance_map = self.model(image_left, image_right_flipped)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")
            target_distmap = F.interpolate(target_map, size=(distance_map.shape[-2], distance_map.shape[-1]), mode="area")

        bce_loss = self.loss_function(probability_map, target_probmap)
        contrastive_loss = self.contrastive_loss_function(distance_map, target_distmap.squeeze())     
        loss = bce_loss + self.lmbda * contrastive_loss
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("loss/train", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        
        probability_map, distance_map = self.model(image_left, image_right_flipped)

        self.batch_size = len(img)

        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")
            target_distmap = F.interpolate(target_map, size=(distance_map.shape[-2], distance_map.shape[-1]), mode="area")

        bce_loss = self.loss_function(probability_map, target_probmap)
        contrastive_loss = self.contrastive_loss_function(distance_map, target_distmap.squeeze())
        loss = bce_loss + torch.mul(self.lmbda, contrastive_loss)
        
        self.log("loss/validation", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.val_pred_labels = torch.hstack((self.val_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.val_target_labels = torch.hstack((self.val_target_labels, torch.where(torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0] > 0.5, 1.0, 0.0)))
            self.logged_images = {"target_probmap":target_map.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu()),
                                  "distance_map":distance_map.detach().cpu()}
        return loss
    
    def test_step(self, batch, batch_idx):
        img, flipped_img, target_map = batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        
        probability_map, distance_map = self.model(image_left, image_right_flipped)
        
        self.batch_size = len(img)
        
        with torch.no_grad():
            target_probmap = F.interpolate(target_map, size=(probability_map.shape[-2], probability_map.shape[-1]), mode="area")
            target_distmap = F.interpolate(target_map, size=(distance_map.shape[-2], distance_map.shape[-1]), mode="area")
        
        bce_loss = self.loss_function(probability_map, target_probmap)
        contrastive_loss = self.contrastive_loss_function(distance_map, target_distmap.squeeze())
        loss = bce_loss + torch.mul(self.lmbda, contrastive_loss)
        
        self.log("loss/test", loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        with torch.no_grad():
            self.test_pred_labels = torch.hstack((self.test_pred_labels, torch.max(torch.sigmoid(probability_map.detach().cpu()).reshape(len(probability_map), -1), dim=1)[0]))
            self.test_target_labels = torch.hstack((self.test_target_labels, torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0]))
            self.logged_images = {"target_probmap":target_probmap.detach().cpu(),
                                  "probability_map": torch.sigmoid(probability_map.detach().cpu()),
                                  "distance_map":distance_map.detach().cpu()}
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
    
if __name__ == "__main__":
       
    x = torch.rand(4, 3, 1024, 1024)
    model = SiameseNetwork()
    probability_map, distance_map = model(x, x)
    print("probability_map.shape: {} \ndistance_map.shape: {}".format(probability_map.shape, distance_map.shape))
    
    loss_function = ContrastiveLoss(margin=5.0)
    contrastive_loss_function = BCELoss()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-5)
    
    classifier = Classifier(model, 
                            optimizer,
                            margin=0.5,
                            lmbda=1.0)
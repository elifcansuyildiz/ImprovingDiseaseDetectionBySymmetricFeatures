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

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
import yaml
import os
import argparse
from typing import Dict
from medai.data.datasets import ChestXDetDataset
import medai.config as config

class ChestDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, batch_size: int = 4, num_workers: int = 8, train_val_split=[0.8, 0.2]):
        super().__init__()
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.train_val_dataset = None
        self.train_dataset, self.val_dataset= None, None
        self.test_dataset = None

    def setup(self, stage:str, **params):        
        """Assign train/val datasets for use in dataloaders. Used dataset is ChestXDetDataset.

        Args:
            stage (str): "fit" or "test"
        """
        if stage == "fit":
            self.train_val_dataset = self.dataset_class(**params)
            lengths = [int(len(self.train_val_dataset) * self.train_val_split[0]), len(self.train_val_dataset) - int(len(self.train_val_dataset) * self.train_val_split[0])]
            self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
        
        elif stage == "test":
            self.test_dataset = self.dataset_class(**params)
    
    @property    
    def train_dataloader(self):
        if self.train_val_dataset.model_name in {"sasn_vanilla", "sasn_split", "baseline"}:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
            #return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn2, num_workers=16, pin_memory=True)
            #return DataLoader(self.train_dataset, sampler=ImbalancedDatasetSampler(self.train_dataset), batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=16, pin_memory=True)        
        elif self.train_val_dataset.model_name == "Mask R-CNN":
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)

    @property 
    def val_dataloader(self):
        if self.train_val_dataset.model_name in {"sasn_vanilla", "sasn_split", "baseline"}:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
            #return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True)
        elif self.train_val_dataset.model_name == "Mask R-CNN":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
            
    @property 
    def test_dataloader(self):
        if self.test_dataset.model_name in {"sasn_vanilla", "sasn_split", "baseline"}:
            return DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
        elif self.test_dataset.model_name in {"Mask R-CNN", "Mask R-CNN-test"}:
            return DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
            
    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
    
def collate_fn(batch):
    return tuple(zip(*batch))

def collate_fn2(batch):
    data_list, flipped_data_list, label_list, bbox_list, target_heatmap = [], [], [], [], []
    
    max_amount = 0
    for i in range(len(batch)):
        if len(batch[i][3])>max_amount:
            max_amount =  len(batch[i][3])
    #print("max amount: ", max_amount, end=" ")        
    for _data, _flipped_data, _label, _bbox, _t_heatmap in batch:
        data_list.append(_data)
        flipped_data_list.append(_flipped_data)
        label_list.append(_label)
        target_heatmap.append(_t_heatmap)
        for i in range(max_amount):
            if i < len(_bbox):
                bbox_list.append(_bbox[i])
            else:
                bbox_list.append( ("None", [-1,-1,-1,-1]) )
            
    return torch.stack(data_list), torch.stack(flipped_data_list), label_list, bbox_list, torch.stack(target_heatmap)
      
def main(params):
    """To check if the ChestXDetDataset is loaded properly.

    Args:
        params (Dict): parameters of ChestXDetDataset
    """
    dm = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    dm.setup(stage="fit", **params["dataset"]["train"])
    loader = dm.train_dataloader
    print(len(loader))

    for idx, (img, img_f, target_heatmap) in enumerate(loader):
        print("image.shape: {} \ntarget_heatmap.shape: {}\n".format(img.shape, target_heatmap.shape))
        
        if idx==4:
            break
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog = 'Disease detection training',
                                     description = 'Trains and evaluates deep learning models',)
    parser.add_argument("--config", "-c", required=True)
    #args = parser.parse_args()
    args = parser.parse_args("--config config.yaml".split())
    config_path = os.path.join( config.CONFIG_DIR, args.config)
    
    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
    
    main(params)
    
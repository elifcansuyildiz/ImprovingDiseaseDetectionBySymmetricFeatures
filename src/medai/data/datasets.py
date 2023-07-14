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

import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PIL import Image
import scipy.io
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from medai.config import tqdm
from typing import Any, Dict, List, Tuple
import multiprocessing as mp

from medai.utils.transforms import Transform, BaseTransform, ColorJitterRandomSizedCrop, Resize, HorizontalFlip
from medai.utils.chest_alignment import AffineTransform

import pytorch_lightning as pl
import medai.config as config

from medai.config import logger

ImagePath = str

class BaseDataset(Dataset):
    def __init__(self):
        self.base_transform = Resize((1024,1024))
        self.horizontal_flip = HorizontalFlip()
        self.random_transform = None
        self.dataset_length = 0
        self.model_name = "sasn_vanilla"

    def __getitem__(self, index) -> Tuple[List[torch.Tensor], List[Any]]:
        if index >= len(self):
            raise IndexError
        
        #image, label, bbox_label, target_heatmap = self.data_reader(index)
        #image, labels, encoded_label, bbox_labels, polygons, target_heatmap = self.data_reader(index)
        sample = self.data_reader(index)
        image, labels, encoded_labels, bboxes, polygons, target_heatmap = sample
        
        if self.model_name in {"sasn_vanilla", "sasn_split", "baseline"}: 
            target_heatmap = target_heatmap.astype(np.float32)
            
            image = self.base_transform(image)
            target_heatmap = self.base_transform(target_heatmap)

            image = torch.vstack((image,image,image))
            
            with torch.no_grad():
                if self.apply_alignment:
                    params = self.alignment_params[index]
                    theta, scale, tx, ty = params["theta"], params["scale"], params["tx"], params["ty"]
                    af = AffineTransform(sx=float(scale), sy=float(scale), theta=float(theta), tx=float(tx), ty=float(ty))
                    image = af(image.unsqueeze(0)).squeeze()
                    target_heatmap = af(target_heatmap.unsqueeze(0)).squeeze().unsqueeze(0)
            
            if self.random_transform is not None and self.random_transform != "None":
                if type(self.random_transform).__name__ == "ChexNetAugmentationMultiImages":
                    [image, target_heatmap] = self.random_transform([image, target_heatmap])
                else:
                    image = self.random_transform(image)

            return image, self.horizontal_flip(image), target_heatmap 
        
        elif self.model_name in {"Mask R-CNN", "Mask R-CNN-test"}:
            image = transforms.ToTensor()(image)  # dtype = torch.float32
            image = torch.vstack((image,image,image)) # shape = (3, 1024, 1024)
            target_heatmap = torch.as_tensor(target_heatmap, dtype=torch.uint8)

            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(encoded_labels, dtype=torch.int64)
            masks = torch.as_tensor(target_heatmap, dtype=torch.uint8)
            
            image = Resize((512,512))(image)
            resized_masks = []
            for mask in masks:
                resized_masks.append(Resize((512,512))(mask))
            masks = torch.stack(resized_masks)
            boxes = boxes / 2
            
            target = {"boxes":boxes, "labels":labels, "masks":masks}
            return image, target

    def __len__(self):
        return self.dataset_length
    
    def data_reader(self, idx):
        raise NotImplementedError
    

class ChestXDetDataset(BaseDataset):
    def __init__(self, 
                 data_dir,
                 json_file_path=None,
                 alignment_csv_path=None,
                 transform=None,
                 img_resize:int=512,
                 binary_labeling:bool=True,
                 ignored_labels:List[str]=[],
                 is_balanced_dataset:bool=False,
                 healthy_CXR_data_dir=None,
                 apply_alignment=False,
                 in_memory:bool=False,
                 num_workers:int=1,
                 model_name:str="sasn_vanilla"):
        
        super().__init__()
        self.base_transform = Resize((img_resize,img_resize))
        self.random_transform = transform
        self.data_dir = os.path.join(config.DATA_DIR, data_dir)
        self.json_file_path = os.path.join(config.DATA_DIR, json_file_path)
        self.alignment_csv_path = os.path.join(config.BASE_DIR, alignment_csv_path)
        self.binary_labeling = binary_labeling
        self.is_balanced_dataset = is_balanced_dataset
        self.healthy_CXR_data_dir = os.path.join(config.DATA_DIR, healthy_CXR_data_dir)
        self.apply_alignment = apply_alignment
        self.alignment_params = []
        self.in_memory = in_memory 
        self.model_name = model_name
        
        with open(self.json_file_path, 'r') as j:
            contents = json.loads(j.read())
        
        self.image_file_paths, self.labels, self.bboxes, self.polygons = [], [], [], []
        for data in contents:
            self.image_file_paths.append(self.data_dir + data["file_name"])
            if data["syms"] == []:
                self.labels.append(["No Finding"])
            else:
                self.labels.append(data["syms"])
            self.bboxes.append(data["boxes"])
            self.polygons.append(data["polygons"])
            
        self.dataset_length = len(self.image_file_paths)
        
        if self.in_memory:
            self.index_to_image_path = {idx:path for idx, path in enumerate(self.image_file_paths)}
            self.images = self.read_data_into_memory(self.image_file_paths, num_workers)
        
        if len(ignored_labels)>0:    
            self.remove_ignored_labels(ignored_labels)
        
        # This part removes data with "No Finding" label for Mask R-CNN model. (Since it works with data including bboxes)
        if self.model_name == "Mask R-CNN":
            tmp_img_file_paths, tmp_labels, tmp_bboxes, tmp_polygons = self.image_file_paths.copy(), self.labels.copy(), self.bboxes.copy(), self.polygons.copy()
            self.image_file_paths, self.labels, self.bboxes, self.polygons = [], [], [], []
            for idx, label_list in enumerate(tmp_labels):
                if "No Finding" not in label_list:
                    self.image_file_paths.append(tmp_img_file_paths[idx])
                    self.labels.append(tmp_labels[idx])
                    self.bboxes.append(tmp_bboxes[idx])
                    self.polygons.append(tmp_polygons[idx])
            self.dataset_length = len(self.image_file_paths)
            
        if self.is_balanced_dataset and self.model_name != "Mask R-CNN":
            self.make_dataset_balanced()
            self.dataset_length = len(self.image_file_paths)
            
        if self.apply_alignment:
            self.alignment_params = self.get_alignment_params(self.alignment_csv_path)
            
        #self.exclude_labels_images(excluded_label="Consolidation")
            
    def data_reader(self, idx):
        if self.in_memory:
            image_path = self.index_to_image_path[idx]
            image = self.images[image_path]
        else:
            image_path = self.image_file_paths[idx]
            with Image.open(image_path,"r") as img:
                image = img.convert('L')
        labels = self.labels[idx]
        encoded_labels = [self.label_to_index[label] for label in (labels)]
        bboxes = self.bboxes[idx]
        polygons = self.polygons[idx]
        target_heatmap = self.target_map_from_polygons(image, polygons)
        return image, labels, encoded_labels, bboxes, polygons, target_heatmap
    
    def read_data_into_memory(self, image_paths: List[ImagePath], num_workers: int) -> Dict[ImagePath, Any]:
        def load_data(path: ImagePath) -> Tuple[ImagePath, Any]:
            with Image.open(path,"r") as img:
                image = img.convert('L')
            return path, image

        with mp.pool.ThreadPool(num_workers) as p:
            result = list(tqdm(p.imap(load_data, image_paths), total=len(image_paths)))

        images: Dict[ImagePath, Any] = {path: image for path, image in result}
        return images
    
    def make_dataset_balanced(self):
        image_paths = glob.glob(self.healthy_CXR_data_dir + "*.png")
        self.image_file_paths = self.image_file_paths + image_paths
        for i in range(len(image_paths)):
            self.labels.append(["No Finding"])
            self.bboxes.append([])
            self.polygons.append([])
            
    def get_alignment_params(self, file_path):
        df = pd.read_csv(file_path)
        alignment_info = df.to_dict("records")

        alignment_params_dict = {}
        for row in alignment_info:
            key = row["file_path"]
            value = {"theta": row["theta"], "scale": row["scale"], "tx": row["tx"], "ty": row["ty"]}
            alignment_params_dict[key] = value

        alignment_params_arr = []
        for file_path in self.image_file_paths:
            short_file_path = "/".join(file_path.split("/")[-3:])
            alignment_params_arr.append(alignment_params_dict[short_file_path])
        return alignment_params_arr
            
    def target_map_from_polygons(self, image, polygons):
        grids = []
        for polygon in polygons:
            nx, ny = image.size
            poly_verts = (polygon)

            # Create vertex coordinates for each grid cell...
            # (<0,0> is at the top left of the grid in this system)
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T

            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny,nx))
            grids.append(grid)

        if self.model_name in {"sasn_vanilla", "sasn_split", "baseline"}:
            segmented = np.zeros_like(image)
            for grid in grids:
                segmented = segmented | grid  # shape: (H, W)
            return segmented
        
        elif self.model_name in {"Mask R-CNN", "Mask R-CNN-test"}:
            if len(grids) == 0:
                masks = torch.zeros(image.size)
                masks = masks.unsqueeze(dim=0)
            else:
                masks = np.stack(grids)     # shape: (num_polygons, H, W)
            return masks
            
    def remove_ignored_labels(self, ignored_labels):
        for j, lbls in enumerate(self.labels):
            for ignored_lbl in ignored_labels:
                indexes = [idx for idx, l in enumerate(self.labels[j]) if l == ignored_lbl]
                
                self.labels[j] = [ele for idx, ele in enumerate(self.labels[j]) if idx not in indexes]
                self.bboxes[j] = [ele for idx, ele in enumerate(self.bboxes[j]) if idx not in indexes] 
                self.polygons[j] = [ele for idx, ele in enumerate(self.polygons[j]) if idx not in indexes]
            
                if self.labels[j] == []:
                    self.labels[j].append("No Finding")
    
    def exclude_labels_images(self, excluded_label):
        tmp_img_file_paths, tmp_labels, tmp_bboxes, tmp_polygons = self.image_file_paths.copy(), self.labels.copy(), self.bboxes.copy(), self.polygons.copy()
        self.image_file_paths, self.labels, self.bboxes, self.polygons = [], [], [], []
        for idx, label_list in enumerate(tmp_labels):
            if excluded_label not in label_list:
                self.image_file_paths.append(tmp_img_file_paths[idx])
                self.labels.append(tmp_labels[idx])
                self.bboxes.append(tmp_bboxes[idx])
                self.polygons.append(tmp_polygons[idx])
        self.dataset_length = len(self.image_file_paths)
    
    @property
    def _all_labels(self):
        """Returns all the labels seen in the images.

        Returns:
            Dict: all unique labels
        """
        label_set = set()
        for lbls in self.labels:
            unique_labels = (set(lbls))
            for u_l in unique_labels:
                label_set.add(u_l)
        return label_set
    
    @property
    def label_to_index(self):
        """Returns the encodings of each label considering the task is binary classification.

        Returns:
            Dict: Encodings of each label.
        """
        if self.binary_labeling:
            label_to_index = {label:1 for label in self._all_labels if label!="No Finding"}
            label_to_index["No Finding"] = 0
        else:
            label_to_index = {label:idx for idx,label in enumerate(self._all_labels)}
        return label_to_index
        
    @property
    def num_of_all_labels_per_disease(self):
        """Returns the number of polygons/bounding boxes per disease.

        Returns:
            Dict: count of each polygons/bounding boxes in images.
        """
        label_counts = {}
        for lbls in self.labels:
            for l in lbls:
                label_counts[l] = label_counts.get(l,0) + 1
        return label_counts
    
    @property
    def num_of_unique_image_labels(self):
        """Returns the number of labels per disease by counting a label ONCE per image
        even if that labels has multiple bounding box/polygon labels in an image.

        Returns:
            Dict: count of each label occuring in images.
        """
        label_counts = {}
        for lbls in self.labels:
            unique_labels = (set(lbls))
            for l in unique_labels:
                label_counts[l] = label_counts.get(l,0) + 1
        return label_counts
    
    @property
    def num_binary_labels(self):
        """Calculates the number of diseased and healthy chest X-ray images. 
        One image can be multilabeled. 
        
        Returns:
            Dict: number of binary labels.
        """
        binary_label_counts = {}
        for lbls in self.labels:
            if lbls[0] == "No Finding":
                binary_label_counts["Healthy"] = binary_label_counts.get("Healthy", 0) + 1
            else:
                binary_label_counts["Diseased"] = binary_label_counts.get("Diseased", 0) + 1
        return binary_label_counts

def main(params):
    """Returns the number of labels and bounding box/polygon annotations of ChestX-Det dataset.

    Args:
        params (Dict): parameters from 'dataset_configs.yaml'
    """    
    train_img_dir = params["dataset"]["train"]["data_dir"]
    json_train_file_path = params["dataset"]["train"]["json_file_path"]
    train_healthy_CXR_data_dir = params["dataset"]["train"]["healthy_CXR_data_dir"]
    train_alignment_csv_path = params["dataset"]["train"]["alignment_csv_path"]
    
    dataset = ChestXDetDataset(data_dir=train_img_dir,
                               json_file_path=json_train_file_path,
                               healthy_CXR_data_dir=train_healthy_CXR_data_dir,
                               alignment_csv_path=train_alignment_csv_path,
                               in_memory=False)
    
    print("Train data all labels: \n{}\n".format(dataset._all_labels))
    print("Label to index: \n{}\n".format(dataset.label_to_index))
    print("The number of bbox/polygon labels per disease: \n{} \n Total count:{}\n".format(dataset.num_of_all_labels_per_disease, sum(dataset.num_of_all_labels_per_disease.values())))
    print("The number of image labels: \n{} \nTotal count: {}\n".format(dataset.num_of_unique_image_labels, sum(dataset.num_of_unique_image_labels.values())))
    print("The number of healthy/diseased image: \n{}\n".format(dataset.num_binary_labels))
    print("--------------------------------------------------------------\n")
    
    test_img_dir = params["dataset"]["test"]["data_dir"]
    json_test_file_path = params["dataset"]["test"]["json_file_path"]
    test_healthy_CXR_data_dir = params["dataset"]["test"]["healthy_CXR_data_dir"]
    test_alignment_csv_path = params["dataset"]["test"]["alignment_csv_path"]
    
    dataset = ChestXDetDataset(data_dir=test_img_dir,
                               json_file_path=json_test_file_path,
                               healthy_CXR_data_dir=test_healthy_CXR_data_dir,
                               alignment_csv_path=test_alignment_csv_path,)
    
    print("Test data all labels: \n{}\n".format(dataset._all_labels))
    print("Label to index: \n{}\n".format(dataset.label_to_index))
    print("The number of bbox/polygon labels per disease: \n{} \n Total count:{}\n".format(dataset.num_of_all_labels_per_disease, sum(dataset.num_of_all_labels_per_disease.values())))
    print("The number of image labels: \n{} \nTotal count: {}\n".format(dataset.num_of_unique_image_labels, sum(dataset.num_of_unique_image_labels.values())))
    print("The number of healthy/diseased image: \n{}".format(dataset.num_binary_labels))
            
if __name__ == "__main__":
    
    yaml_file_dir = os.path.join(config.CONFIG_DIR, "sasn_config.yaml")
    
    with open(yaml_file_dir, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
    
    main(params)
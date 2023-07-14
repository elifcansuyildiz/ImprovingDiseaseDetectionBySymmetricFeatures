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

from medai.data.loader import ChestDataModule
from medai.data.datasets import ChestXDetDataset
import medai.config as config
import numpy as np
import torch
import os
import yaml
from PIL import Image
import copy

def draw_center_vertical_line(img):
    img_c = copy.deepcopy(img)
    cols = img.shape[1]
    img_c[:,cols//2,:] = torch.Tensor([1.0, 0.0, 0.0])
    return img_c

config_file = "config.yaml"
config_file_path = os.path.join(config.CONFIG_DIR, config_file)
with open(config_file_path, "r") as stream:
    params = yaml.safe_load(stream)

#params["dataset"]["train"]["transform"] = None
#params["dataset"]["train"]["apply_alignment"] = True
#dataset_w_a = ChestXDetDataset(**params["dataset"]["train"])

params["dataset"]["test"]["transform"] = None
params["dataset"]["test"]["apply_alignment"] = True
dataset_w_a = ChestXDetDataset(**params["dataset"]["test"])
print(len(dataset_w_a))

save_dir_name = "test_aligned_data"
if not os.path.isdir(save_dir_name):
    os.mkdir(save_dir_name)

for idx in range(len(dataset_w_a.image_file_paths)):
    image_a, image_f_a, target_a = dataset_w_a[idx]
    img_to_save = draw_center_vertical_line(image_a.detach().permute(1,2,0))
    img_to_save = np.array(img_to_save) * 255
    img_to_save = img_to_save.astype(np.uint8)
    im = Image.fromarray(img_to_save)
    im.save(save_dir_name + f"/{idx}.png")
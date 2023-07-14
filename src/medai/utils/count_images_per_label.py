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

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from medai.utils.transforms import Resize
from PIL import Image
from matplotlib.path import Path
import seaborn as sns
import yaml
import os
import medai.config as config

from medai.data.loader import ChestDataModule
from medai.data.datasets import ChestXDetDataset
from medai.utils.visualizer import draw_polygons, draw_bounding_boxes, inspect_label, show_multilabel_images


config_file = "config.yaml"
config_file_path = os.path.join(config.CONFIG_DIR, config_file)
with open(config_file_path, "r") as stream:
    params = yaml.safe_load(stream)
    
dataset = ChestXDetDataset(**params["dataset"]["train"])
test_dataset = ChestXDetDataset(**params["dataset"]["test"])

debug = False

for label in test_dataset._all_labels:
    counter = 0
    for j, lbls in enumerate(test_dataset.labels):
        lbls_set = set(lbls)
        if label in lbls_set:
            if debug:
                print(lbls)
            if len(lbls_set) == 1:
                counter += 1
                if debug:
                    print("heyy")     
    print(f"{label} -> {counter}")
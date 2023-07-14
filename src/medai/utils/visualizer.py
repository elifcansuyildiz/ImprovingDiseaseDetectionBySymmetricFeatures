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

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import PIL
from matplotlib.patches import Polygon, Rectangle
from typing import List
from torchvision import transforms
from torch.utils.data import DataLoader

import logging
import coloredlogs


def show_grid_(dataset):
    """
    Imshow for Tensor.

    Args:
        dataset: Give dataset not dataloader.
    """
    
    plt.figure(figsize=(8*2, 4*2))
    max_i = 32
    for i, sample in enumerate(dataset):
        if i >= max_i:
            break
        image, flipped_image, label, bbox_label = sample
        
        # if labels are encoded and dataset class has decoding dictionary
        """try:
            label = dataset.index_to_label[label]
        except:
            logging.warning("dataset class does not have index_to_label dictionary")
            #raise("dataset class does not have index_to_label dictionary")"""
        
        plt.subplot(4,8,i+1)

        if isinstance(image, PIL.Image.Image):
            plt.imshow(image)
        elif isinstance(image, torch.Tensor):
            plt.imshow(image.numpy().transpose((1, 2, 0)))
        else:
            print("Unknown datatype:", type(image))
            break

        plt.axis("off")
        plt.title(label)

    plt.tight_layout()
    
    
def show_grid(dataset, num_rows=4, num_cols=8):
    """
    Imshow for Tensor.

    Args:
        dataset: Give dataset not dataloader.
    """
    
    fig, axs = plt.subplots(num_rows,num_cols,figsize=(num_cols*4, num_rows*4))
    axs = axs.flatten()

    max_i = num_rows * num_cols
    for i in range(max_i):

        image, encoded_labels, bbox_label, target_heatmap = dataset.data_reader(i)                
        # if labels are encoded and dataset class has decoding dictionary
        try:
            if isinstance(encoded_labels, List):
                labels = []
                for lbl in encoded_labels:
                    labels.append(dataset.index_to_label[lbl])
        except:
            logging.warning("dataset class does not have index_to_label dictionary")
            #raise("dataset class does not have index_to_label dictionary")

        if isinstance(image, PIL.Image.Image):
            transform = transforms.ToTensor()
            image = transform(image)
            image = torch.vstack((image,image,image))
            axs[i].imshow(image.permute((1, 2, 0)))
            
        elif isinstance(image, torch.Tensor):
            axs[i].imshow(image.permute((1, 2, 0)))
        else:
            print("Unknown datatype:", type(image))
            break
        
        axs[i].axis("off")
        if len(bbox_label)>0:
            bbox_lbls = [i[0] for i in bbox_label]
            axs[i].set_title(bbox_lbls)
        else:
            axs[i].set_title(labels)
            
        if len(bbox_label)>0:
            for value in bbox_label:
                
                x_ = value[1][0]
                y_ = value[1][1]
                w = value[1][2]
                h = value[1][3]
                #print(x_, y_, w, h)
                
                y = y_
                x = x_
                axs[i].add_patch(Rectangle( xy=(x,y), width=w, height=h, fill=False ))
                #axs[i].add_patch(Rectangle( xy=(5,200), width=200, height=200, fill=False ))
                axs[i].text(x, y, value[0], color="blue")

    plt.tight_layout()
    
def show_image_target_heatmap(dataset, num_image=4, heatmap_type="rectangle"):
    fig, axs = plt.subplots(num_image, 2, figsize=(8, num_image*4))
    axs = np.array(axs).reshape((num_image, 2))
    
    dataset.heatmap_type = heatmap_type
    
    for i in range(num_image):
        image, label, bbox_label, target_heatmap = dataset.data_reader(i)
        
        if len(bbox_label)==0:
            continue
        
        for value in bbox_label:      
            x_ = value[1][0]
            y_ = value[1][1]
            w = value[1][2]
            h = value[1][3]
            y = y_
            x = x_
            
            if isinstance(image, PIL.Image.Image):
                transform = transforms.ToTensor()
                image = transform(image)
                image = torch.vstack((image,image,image))
            
            axs[i][0].imshow(image.permute((1, 2, 0)))
            axs[i][0].add_patch(Rectangle( xy=(x,y), width=w, height=h, fill=False ))
            axs[i][0].text(x, y, value[0], color="blue")
            
            axs[i][1].imshow(target_heatmap)
        
def draw_polygons(samples, axes):
    """Draws polygons for each disease observations to the given axes.

    Args:
        samples (List): Dataset samples including the list of (image, labels, encoded_label, bboxes, polygons).
        axes (_type_): List of axis with the same number of samples.
    """
    for sample,axs in zip(samples, axes):
        image, labels, encoded_label, bboxes, polygons, target_heatmap = sample
        
        #fig, axs = plt.subplots(1,1)
        axs.imshow(image, cmap="gray")

        if labels[0] == "No Finding":
            title="No Finding"
            
        for polygon,label in zip(polygons, labels):
            p = Polygon(np.array(polygon), fill=False, color="blue")
            axs.add_patch(p)
            center_coord = np.mean(np.array(polygon), axis=0)
            axs.text(center_coord[0], center_coord[1], label, horizontalalignment="center")
            title=str(set(labels))
                
        axs.set_title(title)
        
def draw_bounding_boxes(samples, axes):
    """Draws bounding boxes for each disease observations to the given axes.

    Args:
        samples (List): Dataset samples including the list of (image, labels, encoded_label, bboxes, polygons).
        axes (_type_): List of axis with the same number of samples.
    """
    for sample,axs in zip(samples, axes):
        image, labels, encoded_label, bboxes, polygons, target_heatmap = sample
        
        #fig, axs = plt.subplots(1,1)
        axs.imshow(image, cmap="gray")
        
        for bbox,label in zip(bboxes, labels):
            x1,y1,x2,y2 = bbox
            axs.add_patch(Rectangle( xy=(x1,y1), width=x2-x1, height=y2-y1, fill=False, color="red" ))
            
def show_multilabel_images(dataset, save_dir="outputs", data_range=(20,44), num_rows_cols=(6,4), figsize=(30,40)):
    """Outputs and saves the figure of multiple images with polygon annotations drawn for each disease observations.

    Args:
        dataset 
        save_dir (str, optional): Path to save the output. Defaults to "outputs".
        data_range (tuple, optional): The range of data indexes. Defaults to (20,44).
        num_rows_cols (tuple, optional): Number of data in (row, col). This input should match data_range Defaults to (6,4).
        figsize (tuple, optional): The size of the figure. Defaults to (30,40).
    """
    fig, axs = plt.subplots(num_rows_cols[0], num_rows_cols[1], figsize=figsize)
    axs = axs.flatten()
    
    idx = 0
    for i in range(*data_range):
        image, labels, encoded_label, bboxes, polygons, target_heatmap = dataset.data_reader(i)
        
        #fig, axs = plt.subplots(1,1)
        axs[idx].imshow(image, cmap="gray")
        axs[idx].axis("off")

        if labels[0] == "No Finding":
            title="No Finding"
            
        for polygon,label in zip(polygons, labels):
            p = Polygon(np.array(polygon), fill=False, color="red")
            axs[idx].add_patch(p)
            center_coord = np.mean(np.array(polygon), axis=0)
            axs[idx].text(center_coord[0], center_coord[1], label, horizontalalignment="center")
            title=str(set(labels))
                
        axs[idx].set_title(title)
        idx += 1
        
    fig.tight_layout()    
    fig.savefig(save_dir + "/multilabel_images.png")
    print("Saved ")
    
           
def inspect_label(dataset, label, save_dir="outputs", num_rows_cols=(12,10), figsize=(30,40)):
    """Outputs and saves the figure of multiple images including only the specified label to inspect label characteristics.

    Args:
        dataset
        label (str): specified label to inspect
        save_dir (str, optional): Path to save the output. Defaults to "outputs".
        num_rows_cols (tuple, optional): the number of images in (row, col). Defaults to (12,10).
        figsize (tuple, optional): size of the figure. Defaults to (30,40).
    """
    fig, axs = plt.subplots(num_rows_cols[0], num_rows_cols[1], figsize=figsize)
    axs = axs.flatten()

    idx = 0
    for i in range(len(dataset)):
        image, labels, encoded_labels, bboxes, polygons, target_heatmap = dataset.data_reader(i)
        
        if label not in labels:
            continue
        
        for j,lbl in enumerate(labels):
            if lbl == label:
                #axs[idx].imshow(target_heatmap); axs[idx].axis("off")
                axs[idx].imshow(image, cmap="gray"); axs[idx].axis("off")
                p = Polygon(np.array(polygons[j]), fill=False, color="blue")
                axs[idx].add_patch(p)
                center_coord = np.mean(np.array(polygons[j]), axis=0)
                #axs[idx].text(center_coord[0], center_coord[1], label, horizontalalignment="center")
                axs[idx].set_title(label)
        idx += 1
            
        if idx == (num_rows_cols[0]*num_rows_cols[1]):
            break
        
    fig.tight_layout()
    fig.savefig(save_dir + '/data_with_label_' + label + ".png")
    print("Saved " + label + " outputs.")
    
def visualize_predictions(model, dataset, process_half_image = False):
    fig, axs = plt.subplots(len(dataset), 4, figsize=(12,len(dataset)*3))
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, sample in enumerate(loader):
        img, flipped_img, target_map = sample
        
        if process_half_image:
            image_left = img[:, :, :, :img.shape[2]//2]
            image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
            with torch.no_grad():
                probability_map, distance_map = model(image_left.to(model.device), image_right_flipped.to(model.device))
                probability_map = torch.sigmoid(probability_map)
        else:
            with torch.no_grad():
                probability_map, distance_map = model(img.to(model.device), flipped_img.to(model.device))
                probability_map = torch.sigmoid(probability_map)
        #print("probability_map.shape: {} \ndistance_map.shape: {}".format(probability_map.shape, distance_map.shape))
        axs[idx][0].imshow(img[0].permute(1,2,0))
        axs[idx][1].imshow(target_map[0].permute(1,2,0))
        axs[idx][2].imshow(probability_map[0].detach().cpu().permute(1,2,0))
        axs[idx][3].imshow(distance_map[0].detach().cpu())
    
    return fig

def visualize_maskrcnn_predictions(model, dataset, device="cuda", show_score=False):
    fig, axs = plt.subplots(len(dataset), 2, figsize=(6,len(dataset)*3))
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, (image, target) in enumerate(loader):
        model.eval()
        with torch.no_grad():
            outputs = model.predict(image.to(device))  

        target_bboxes = target["boxes"][0]
        predicted_bboxes = outputs[0]["boxes"]
        predicted_scores = outputs[0]["scores"]
                
        axs[idx][0].imshow(image[0].permute(1,2,0))
        axs[idx][0].set_title("Ground Truth")
        for box in target_bboxes:
            x1,y1,x2,y2 = box
            axs[idx][0].add_patch(Rectangle( xy=(x1,y1), width=x2-x1, height=y2-y1, fill=False, color="red" ))
            
        axs[idx][1].imshow(image[0].permute(1,2,0))
        axs[idx][1].set_title("Prediction")
        for box,score in zip(predicted_bboxes, predicted_scores):
            x1,y1,x2,y2 = box.detach().cpu().numpy()
            axs[idx][1].add_patch(Rectangle( xy=(x1,y1), width=x2-x1, height=y2-y1, fill=False, color="red" ))
            if show_score:
                axs[idx][1].text((x1+x2)//2, (y1+y2)//2, str(score), horizontalalignment="center")
    
    return fig
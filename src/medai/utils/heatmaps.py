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

import copy
import torchvision
import torch
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from medai.utils.transforms import Resize

def get_images_gtmasks(iterator, num_images, sample_set=None):
    chest_images = []
    gt_masks = []

    for i in tqdm(range(num_images)):
        if sample_set is not None and i not in sample_set:
            sample_batch = next(iterator)
            continue
        #print(i, end=" ")
        
        sample_batch = next(iterator)
        img, flipped_img, target_map = sample_batch
        
        chest_images.append(img)
        gt_masks.append(target_map)
    
    return torch.stack(chest_images), torch.stack(gt_masks)

def get_images_polygon_labels(dataset, sample_range=(0,16), sample_set=None):
    images = []
    polygon_labels = []
    
    for idx in range(*sample_range):
        if sample_set is not None and idx not in sample_set:
            continue
        #print(idx, end=" ")
        
        sample = dataset.data_reader(idx)
        image, labels, encoded_label, bboxes, polygons, target_heatmap = sample
        image = Resize((512,512))(image)
        image = torch.vstack((image,image,image))
        images.append(image)
        
        resized_polygons = copy.deepcopy(polygons)
        for i in range(len(polygons)):
            for j in range(len(polygons[i])):
                resized_polygons[i][j][0] = polygons[i][j][0] // 2
                resized_polygons[i][j][1] = polygons[i][j][1] // 2
        
        polygon_labels.append(resized_polygons)
        
    return torch.stack(images), polygon_labels

def get_SASN_vanilla_heatmaps(model, iterator, num_images, to_overlay=False, sample_set=None):
    prob_heatmaps = []
    distance_maps = []

    model.eval()
    for i in tqdm(range(num_images)):
        if sample_set is not None and i not in sample_set:
            sample_batch = next(iterator)
            continue
        #print("i: ", i, end=" ")
        
        sample_batch = next(iterator)
        img, flipped_img, target_map = sample_batch
        
        with torch.no_grad():
            probability_map, distance_map = model(img.to(model.device), flipped_img.to(model.device))
            probability_map = torch.sigmoid(probability_map)

        if to_overlay:
            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(img[0]), to_pil_image(probability_map[0].squeeze(0).detach().numpy(), mode='F'), alpha=0.5)
            result = torchvision.transforms.ToTensor()(result)
            prob_heatmaps.append(result)
            
            distance_map_normalized = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())
            result = overlay_mask(to_pil_image(img[0]), to_pil_image(distance_map_normalized.squeeze(0).detach().numpy(), mode='F'), alpha=0.5)
            result = torchvision.transforms.ToTensor()(result)
            distance_maps.append(result)
            
        else:
            prob_heatmaps.append(probability_map)
            distance_maps.append(distance_map)
        
    return torch.stack(prob_heatmaps), torch.stack(distance_maps)


def get_SASN_split_heatmaps(model, iterator, num_images, to_overlay=False, sample_set=None):
    prob_heatmaps = []
    distance_maps = []

    model.eval()
    for i in tqdm(range(num_images)):
        if sample_set is not None and i not in sample_set:
            sample_batch = next(iterator)
            continue
        #print("i: ", i, end=" ")
        
        sample_batch = next(iterator)
        img, flipped_img, target_map = sample_batch
        
        image_left = img[:, :, :, :img.shape[2]//2]
        image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
        with torch.no_grad():
            probability_map, distance_map = model(image_left.to(model.device), image_right_flipped.to(model.device))
            probability_map = torch.sigmoid(probability_map)

        if to_overlay:
            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(img[0]), to_pil_image(probability_map[0].squeeze(0).detach().numpy(), mode='F'), alpha=0.5)
            result = torchvision.transforms.ToTensor()(result)
            prob_heatmaps.append(result)
            
            distance_map_normalized = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())
            result = overlay_mask(to_pil_image(img[0]), to_pil_image(distance_map_normalized.squeeze(0).detach().numpy(), mode='F'), alpha=0.5)
            result = torchvision.transforms.ToTensor()(result)
            distance_maps.append(result)
            
        else:
            prob_heatmaps.append(probability_map)
            distance_maps.append(distance_map)
        
    return torch.stack(prob_heatmaps), torch.stack(distance_maps)

def get_CheXNet_heatmaps(model, iterator, num_images, sample_set=None):
    heatmaps = []
    
    model.eval()
    # Run this before feeding input to the model. 
    cam_extractor = GradCAMpp(model, target_layer=model.densenet.features.denseblock4)

    for i in tqdm(range(num_images)):
        if sample_set is not None and i not in sample_set:
            sample_batch = next(iterator)
            continue
        #print("i: ", i, end=" ")
        
        sample_batch = next(iterator)
        preprocessed_img, flipped_img, target_map = sample_batch

        # Preprocess your data and feed it to the model
        out = model(preprocessed_img)

        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(class_idx=0, scores=out, normalized=True)
        
        activation_map[0] *= torch.sigmoid(out.detach())

        # Resize the CAM and overlay it
        preprocessed_img_normalized = (preprocessed_img - preprocessed_img.min()) / (preprocessed_img.max() - preprocessed_img.min())
        result = overlay_mask(to_pil_image(preprocessed_img_normalized[0]), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        result = torchvision.transforms.ToTensor()(result)

        heatmaps.append(result)
        
    # Once you're finished, clear the hooks on your model
    cam_extractor.remove_hooks()
    
    return torch.stack(heatmaps)

def get_MaskRCNN_heatmaps(model, iterator, num_images, print_scores=False, sample_set=None):
    masks_list = []
    boxes_list = []

    model.eval()
    for i in tqdm(range(num_images)):
        if sample_set is not None and i not in sample_set:
            sample_batch = next(iterator)
            continue
        #print("i: ", i, end=" ")
        
        sample_batch = next(iterator)
        image, target = sample_batch
        
        with torch.no_grad():
            outputs = model.predict(image)

        masks = outputs[0]["masks"] 
        boxes = outputs[0]["boxes"] 
        scores = outputs[0]["scores"]
        #gt_masks = target["masks"]
        if print_scores:
            print(scores)
        #print(masks.shape[0])
        
        if masks.shape[0] == 0:
            pred_mask = torch.zeros((1,512,512))
        else:
            pred_mask = torch.max(masks, dim=0).values
            
        masks_list.append(pred_mask)
        boxes_list.append(boxes)
        
    return torch.stack(masks_list), boxes_list
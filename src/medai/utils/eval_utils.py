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

import pickle
import collections
import torch
import numpy as np
import pandas as pd
import os
import cv2
from torchmetrics.wrappers.bootstrapping import BootStrapper
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from medai.utils.metrics import BinaryMetric
from medai.utils.transforms import Resize
from medai.utils.localization import non_max_suppression, mask_to_bbox_with_connected_components


def load_model(model, path):
    checkpoint = torch.load(path)
    print(checkpoint.keys())
    
    test_state = checkpoint["state_dict"]
    new_state = collections.OrderedDict()

    for key, value in test_state.items():
        new_state[key[6:]] = value

    model.load_state_dict(new_state)
    
    return model

def load_chexnet_model(model, path):
    checkpoint = torch.load(path)
    print(checkpoint.keys())
    
    test_state = checkpoint["state_dict"]
    new_state = collections.OrderedDict()

    for key, value in test_state.items():
        if "model." in key:
            new_state[key[6:]] = value

    model.load_state_dict(new_state)
    
    return model

def get_preds_labels(model, dataloader, model_name="sasn_vanilla", include_contrastive=True):
    pred_labels, target_labels = torch.asarray([]), torch.asarray([])
    model.eval()

    for batch in dataloader:
        img, flipped_img, target_map = batch
        
        if model_name == "sasn_vanilla":
            if include_contrastive:
                probability_map, distance_map = model(img, flipped_img)
            else:
                probability_map = model(img, flipped_img)
        elif model_name == "sasn_split":
            image_left = img[:, :, :, :img.shape[2]//2]
            image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
            if include_contrastive:
                probability_map, distance_map = model(image_left, image_right_flipped)
            else:
                probability_map = model(image_left, image_right_flipped)
        elif model_name == "basic_model":
            with torch.no_grad():
                probability_map = model(img)
        
        if model_name == "sasn_vanilla" or model_name == "sasn_split" or model_name == "basic_model":
            with torch.no_grad():
                pred_label = torch.max(torch.sigmoid(probability_map).detach().cpu().reshape(len(probability_map), -1), dim=1)[0]
                target_label = torch.where(torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0] > 0.5, 1.0, 0.0)
                pred_labels = torch.hstack((pred_labels, pred_label))
                target_labels = torch.hstack((target_labels, target_label))
        
        elif model_name == "chexnet":
            with torch.no_grad():
                pred_label = model(img)
                pred_label = torch.sigmoid(pred_label).flatten()
                target_label = torch.where(torch.max(target_map.detach().cpu().reshape(len(target_map), -1), dim=1)[0] > 0, 1.0, 0.0).flatten()
                pred_labels = torch.hstack((pred_labels, pred_label))
                target_labels = torch.hstack((target_labels, target_label))

    return pred_labels, target_labels

def get_maskrcnn_pred_gt_labels(model, loader):
    pred_labels = []
    gt_labels = []

    for idx, (image, target) in enumerate(loader):
        model.eval()
        with torch.no_grad():
            outputs = model.predict(image)
            #print(outputs[0]["scores"], " ", outputs[0]["labels"])
            
        labels = outputs[0]["scores"]
        if len(labels)>0:
            pred_labels.append(labels.max())
        else:
            pred_labels.append(torch.Tensor([0.0])[0])
            
        gt_lbls = target[0]["labels"]
        if gt_lbls[0]==0:
            gt_labels.append(0)
        else:
            gt_labels.append(1)
    
    pred_labels = torch.as_tensor(pred_labels, dtype=torch.float32)
    gt_labels = torch.as_tensor(gt_labels, dtype=torch.int32)
    
    return pred_labels, gt_labels

def calculate_metrics(pred_labels, target_labels, threshold=0.5):
    results = {"accuracy": BinaryMetric.accuracy(pred_labels, target_labels, threshold),
               "F1 score": BinaryMetric.f1_score(pred_labels, target_labels, threshold),
               "precision": BinaryMetric.precision(pred_labels, target_labels, threshold),
               "recall": BinaryMetric.recall(pred_labels, target_labels, threshold),
               "specificity": BinaryMetric.specificity(pred_labels, target_labels, threshold),
               "auroc": BinaryMetric.auroc(pred_labels, target_labels),
               "avg precision": BinaryMetric.avg_precision(pred_labels, target_labels)}
    
    return results

def save_experiment_preds_targets(val_pred_labels, val_target_labels, test_pred_labels, test_target_labels, pickle_path="preds_targets.pickle"):
    preds_targets = {"val_pred_labels": val_pred_labels,
                     "val_target_labels": val_target_labels,
                     "test_pred_labels": test_pred_labels,
                     "test_target_labels": test_target_labels}

    with open(pickle_path, 'wb') as file:
        pickle.dump(preds_targets, file, protocol=pickle.HIGHEST_PROTOCOL)
 

def load_experiment_preds_targets(file_name:str) -> dict:
    with open(file_name, 'rb') as file:
        pred_targets = pickle.load(file)
    return pred_targets


def get_scores(model, data_module, model_path, pickle_path, model_name, threshold=0.5, include_sasn_contrastive=True):
    if os.path.isfile(pickle_path):
        preds_targets_dict = load_experiment_preds_targets(file_name=pickle_path)
        #
        val_metric_scores = calculate_metrics(preds_targets_dict["val_pred_labels"], preds_targets_dict["val_target_labels"], threshold)
        #val_metric_scores = None
        test_metric_scores = calculate_metrics(preds_targets_dict["test_pred_labels"], preds_targets_dict["test_target_labels"], threshold)
    else:
        if model_name != "chexnet":
            model = load_model(model=model, path=model_path)
        else:
            model = load_chexnet_model(model=model, path=model_path)
                
        model.eval()

        ##
        val_preds_labels, val_target_labels = get_preds_labels(model, data_module.val_dataloader, model_name=model_name, include_contrastive=include_sasn_contrastive)
        print("Val Lengths: ", len(val_preds_labels), " ", len(val_target_labels))
        val_metric_scores = calculate_metrics(val_preds_labels, val_target_labels, threshold)
        print("val_metric_scores: ", val_metric_scores)
        #val_preds_labels, val_target_labels = None, None
        #val_metric_scores = None
    
        test_preds_labels, test_target_labels = get_preds_labels(model, data_module.test_dataloader, model_name=model_name, include_contrastive=include_sasn_contrastive)
        print("Test Lengths: ", len(test_preds_labels), " ", len(test_target_labels))
        test_metric_scores = calculate_metrics(test_preds_labels, test_target_labels, threshold)
        print("test_metric_scores: ", test_metric_scores)

        save_experiment_preds_targets(val_preds_labels, val_target_labels, test_preds_labels, test_target_labels, pickle_path=pickle_path)
    
    return val_metric_scores, test_metric_scores

def round_value(value):
    return round((float(value)*100),2)

def get_best_threshold(experiment_file_name, threshold_range=(0.1,0.91,0.1)):
    if not os.path.isfile(experiment_file_name):
        return "No such file or directory!!"
        
    preds_targets_dict = load_experiment_preds_targets(file_name=experiment_file_name)
    
    max_val_thresh, max_val_F1score = 0, 0.0
    max_test_thresh, max_test_F1score = 0, 0.0
    
    for thresh in np.arange(*threshold_range):
        thresh = round(float(thresh),2)
        
        val_metric_scores = calculate_metrics(preds_targets_dict["val_pred_labels"], preds_targets_dict["val_target_labels"], threshold=thresh)
                
        if val_metric_scores['F1 score'] > max_val_F1score:
            max_val_F1score = val_metric_scores['F1 score']
            max_val_thresh = thresh
            
        test_metric_scores = calculate_metrics(preds_targets_dict["test_pred_labels"], preds_targets_dict["test_target_labels"], threshold=thresh)
        
        if test_metric_scores['F1 score'] > max_test_F1score:
            max_test_F1score = test_metric_scores['F1 score']
            max_test_thresh = thresh
        
        #print(f"thresh: {thresh} Val F1 score: {val_metric_scores['F1 score']} Test F1 score: {test_metric_scores['F1 score']}")        
        
    return {"val":{"thresh":max_val_thresh, "F1 Score":round_value(max_val_F1score)}, "test":{"thresh":max_test_thresh, "F1 Score":round_value(max_test_F1score)}}

def get_best_threshold_for_balanced_acc(predictions, targets, threshold_range=(0.1,0.91,0.1)):   
    max_thresh, max_balanced_acc = 0, 0.0
    
    for thresh in np.arange(*threshold_range):
        thresh = round(float(thresh),2)
        
        metric_scores = calculate_metrics(predictions, targets, threshold=thresh)
        
        balanced_acc = (metric_scores['recall'] + metric_scores['specificity']) / 2
              
        if balanced_acc > max_balanced_acc:
            max_balanced_acc = balanced_acc
            max_thresh = thresh
        
        print(f"thresh: {thresh} Val Balanced Acc: {balanced_acc}")        
        
    return {"thresh":max_thresh, "Balanced Acc":round_value(max_balanced_acc)}

def get_best_threshold_for_F1(predictions, targets, threshold_range=(0.1,0.91,0.1)):   
    max_thresh, max_F1score = 0, 0.0
    
    for thresh in np.arange(*threshold_range):
        thresh = round(float(thresh),2)
        
        metric_scores = calculate_metrics(predictions, targets, threshold=thresh)
                
        if metric_scores['F1 score'] > max_F1score:
            max_F1score = metric_scores['F1 score']
            max_thresh = thresh
        
        #print(f"thresh: {thresh} Val F1 score: {metric_scores['F1 score']}")        
        
    return {"thresh":max_thresh, "F1 Score":round_value(max_F1score)}


def significance_interval(base_metric, predictions, targets, quantile:float=0.95, num_bootstraps:int=100):
    """Calculates the significance interval of a metric with bootstrapping

    Args:
        base_metric : Evaluation metric
        predictions (torch.Tensor): List of all predictions.
        targets (torch.Tensor): List of all targets.
        quantile (float): For significance interval. Defaults to 0.95.
        num_bootstraps (int): Number of bootstraps. Defaults to 100.

    Returns:
        (Dict, float, float): mean, std, raw values of bootstrapper, lower limit and upper limit of significance interval
    """
    bootstrapper = BootStrapper(base_metric, num_bootstraps=num_bootstraps, raw=True)

    bootstrapper.update(predictions,targets)
    output = bootstrapper.compute()

    lower_limit = np.quantile(output["raw"].numpy(), (1-quantile)/2)
    upper_limit = np.quantile(output["raw"].numpy(), quantile + ((1-quantile)/2))

    return output, lower_limit, upper_limit

def create_table(scores:dict, experiment_name:str, threshold:float=0.5):
    columns = ["experiment", "threshold", "AUROC", "F1 score", "precision", "recall", "specificity", "accuracy", "average precision"]
    row = [experiment_name, threshold, round_value(scores["auroc"]), round_value(scores["F1 score"]), round_value(scores["precision"]), round_value(scores["recall"]), round_value(scores["specificity"]), round_value(scores["accuracy"]), round_value(scores["avg precision"])]
    df = pd.DataFrame(data=[row], columns=columns)
    return df
   
def update_score_table(data_frame, csv_file_path:str):
    if not os.path.isfile(path=csv_file_path):
        df = pd.DataFrame(columns=["experiment", "threshold", "AUROC", "F1 score", "precision", "recall", "specificity", "accuracy", "average precision"])
        df.to_csv(csv_file_path, index=False)
    score_table = pd.read_csv(csv_file_path)
    updated_table = pd.concat([score_table, data_frame])
    updated_table.to_csv(csv_file_path, index=False)
    
def save_score_table(metric_scores:dict,  experiment_name:str, threshold:float=0.5, save_path:str="csv_outputs/scores.csv"):
    df = create_table(metric_scores, experiment_name, threshold=threshold)
    update_score_table(df, csv_file_path=save_path)
    
def get_gtmasks(dataloader, num_images=16):
    target_maps = []

    for i, sample_batch in enumerate(dataloader):
        img, flipped_img, target_map = sample_batch
        
        target_maps.append(target_map)
        
    return target_maps

def get_probmaps(model, dataloader, model_name="sasn_vanilla", num_images=16):
    
    prob_maps = []

    for i, sample_batch in enumerate(dataloader):
        img, flipped_img, target_map = sample_batch
        
        model.eval()
        with torch.no_grad():
            if model_name == "sasn_vanilla":
                probability_map, distance_map = model(img.to(model.device), flipped_img.to(model.device))
            elif model_name == "sasn_split":
                image_left = img[:, :, :, :img.shape[2]//2]
                image_right_flipped = flipped_img[:, :, :, :flipped_img.shape[2]//2]
                probability_map, distance_map = model(image_left.to(model.device), image_right_flipped.to(model.device))
            
            probability_map = torch.sigmoid(probability_map)
            probability_map = Resize((512,512))(probability_map[0])
            prob_maps.append(probability_map)
            
        if i == num_images-1:
            break

    return prob_maps

def probmaps_to_masks(prob_maps, threshold):
    thresholded_probmaps = []

    for p_map in prob_maps:
        thresholded_probmaps.append( torch.where(p_map>threshold, 1, 0) )
        
    return thresholded_probmaps

def get_boundingboxes_from_masks(masks):
    boxes_list = []
    
    for mask in masks:
        mask = mask.squeeze().unsqueeze(dim=2)
        image = mask.cpu().numpy().astype(np.uint8)
        #image = np.moveaxis(image, 0, -1)
        boxes = mask_to_bbox_with_connected_components(image)
        boxes_list.append(boxes)
        
    return boxes_list

def get_gt_bboxes(dataset, sample_range=(0,16)):
    gt_boxes = []
    
    for i in range(*sample_range):
        image, labels, encoded_labels, bboxes, polygons, target_heatmap = dataset.data_reader(i)
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        boxes = boxes / 2
        gt_boxes.append(boxes)
    
    return gt_boxes

def apply_nms(bboxes):
    nms_results = []
    
    for bbox in bboxes:
        nms_results.append(non_max_suppression(bbox.numpy()))
        
    return nms_results

def to_torchmetrics_format(detections, include_scores:bool):
    """Converts list of lists of lists or list of tensor matrices to tensor type.

    Args:
        detections (List[List[List]], List[Tensor[Tensor]]): list of lists of lists or list of tensor matrices
        include_scores (bool): _description_

    Returns:
        dict: boxes, (scores), labels
    """
    out = []
    for detection in detections:
        boxes = []
        scores = []
        labels = []
        for bbox in detection:
            if torch.is_tensor(bbox):
                boxes.append(bbox.detach().cpu().numpy())
            else:
                boxes.append(bbox)
            scores.append(1.0)
            labels.append(0)
        if include_scores:
            out.append(dict(boxes=torch.tensor(boxes),
                            scores=torch.tensor(scores),
                            labels=torch.tensor(labels),
                            )
                    )
        else:
            out.append(dict(boxes=torch.tensor(boxes),
                            labels=torch.tensor(labels),
                            )
                    )
    return out

def calculate_mean_average_precision(predicted_boxes, groundtruth_boxes):
    metric = MeanAveragePrecision()
    metric.update(predicted_boxes, groundtruth_boxes)
    scores = metric.compute()
    return scores

def bbox_to_mask(img_shape, predictions):
    img = np.zeros(img_shape)
    for x1,y1,x2,y2 in predictions:
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), 1.0, -1)
    return torch.tensor(img)

# input two 2D arrays between 0.0 and 1.0. applies a threshold
def pixelwise_comparison_statistics(prediction, ground_truth, threshold=0.5):
    if (torch.sum(prediction.detach().cpu()>1.0) > 0 or torch.sum(ground_truth.detach().cpu()>1.0) > 0):
        raise Exception("input arrays must be between 0.0 and 1.0")
    
    if (len(prediction.shape) != 2 or len(ground_truth.shape) != 2):
        raise Exception("input arrays must be 2D")
    
    if (prediction.shape[0] != ground_truth.shape[0] or prediction.shape[1] != ground_truth.shape[1]):
        raise Exception("input arrays have different shapes")
    
    ground_truth_b = np.array(ground_truth > threshold)
    prediction_b = np.array(prediction > threshold)
    
    out = {}
    out["total_pixels"] = prediction.shape[0] * prediction.shape[1]
    out["true_positive"] = np.count_nonzero(ground_truth_b * prediction_b)
    out["false_positive"] = np.count_nonzero((~ground_truth_b) * prediction_b)
    out["true_negative"] = np.count_nonzero((~ground_truth_b) * (~prediction_b))
    out["false_negative"] = np.count_nonzero(ground_truth_b * (~prediction_b))
    
    if (out["true_positive"]+out["false_negative"]) == 0:
        out["true_positive_rate"] = 0 #np.nan
    else:
        out["true_positive_rate"] = out["true_positive"] / (out["true_positive"] + out["false_negative"])
    
    if (out["false_positive"] + out["true_negative"]) == 0:
        out["false_positive_rate"] = 0 #np.nan
    else:
        out["false_positive_rate"] = out["false_positive"] / (out["false_positive"] + out["true_negative"])
    
    if (out["true_negative"] + out["false_positive"]) == 0:
        out["true_negative_rate"] = 0 #np.nan
    else:
        out["true_negative_rate"] = out["true_negative"] / (out["true_negative"] + out["false_positive"])
    
    if (out["false_negative"] + out["true_positive"]) == 0:
        out["false_negative_rate"] = 0 #np.nan
    else:
        out["false_negative_rate"] = out["false_negative"] / (out["false_negative"] + out["true_positive"])
    
    out["accuracy"] = (out["true_positive"]+out["true_negative"])/out["total_pixels"]
    
    if (out["true_negative"]+out["false_positive"]) == 0:
        out["specificity"] = 0 #np.nan
    else:
        out["specificity"] = out["true_negative"]/(out["true_negative"]+out["false_positive"])
    
    if (out["true_positive"]+out["false_negative"]) == 0:
        out["recall"] = 0 #np.nan
    else:
        out["recall"] = out["true_positive"]/(out["true_positive"]+out["false_negative"])
    
    if (out["true_positive"]+out["false_positive"]) == 0:
        out["precision"] = 0 #np.nan
    else:
        out["precision"] = out["true_positive"]/(out["true_positive"]+out["false_positive"])
    
    if out["recall"] is np.nan or out["precision"] is np.nan or (out["precision"]+out["recall"])==0:
        out["f1_score"] = 0 #np.nan
    else:
        out["f1_score"] = (2*out["precision"]*out["recall"]) / (out["precision"]+out["recall"])
    return out

def calculate_pixelwise_classification_scores(pred_masks, target_masks, to_return_mean=True):
    
    tpr_list, fnr_list, fpr_list, precision_list = [], [], [], []
    num_images = len(pred_masks)
    
    for idx in range(num_images):
        pred_mask = pred_masks[idx].squeeze()
        target_mask = target_masks[idx].squeeze()
        result = pixelwise_comparison_statistics(pred_mask, target_mask)
        tpr_list.append( result["true_positive_rate"] )
        fnr_list.append( result["false_negative_rate"] )
        fpr_list.append( result["false_positive_rate"])
        precision_list.append(result["precision"])
    
    if to_return_mean:
        return {"tpr":np.array(tpr_list).mean(), "fnr":np.array(fnr_list).mean(), "fpr":np.array(fpr_list).mean(), "precision":np.array(precision_list).mean()}
    else:
        return tpr_list, fnr_list, fpr_list, precision_list

def calculate_pixelwise_classification_scores_for_maskrcnn(maskrcnn_boxes, target_masks, to_return_mean=True):
    
    tpr_list, fnr_list, fpr_list, precision_list = [], [], [], []
    num_images = len(target_masks)
    
    for idx in range(num_images):
        pred_mask = bbox_to_mask((512,512), maskrcnn_boxes[idx])
        target_mask = target_masks[idx].squeeze()
        result = pixelwise_comparison_statistics(pred_mask, target_mask)
        tpr_list.append( result["true_positive_rate"] )
        fnr_list.append( result["false_negative_rate"] )
        fpr_list.append( result["false_positive_rate"])
        precision_list.append(result["precision"])
        
    if to_return_mean:
        return {"tpr":np.array(tpr_list).mean(), "fnr":np.array(fnr_list).mean(), "fpr":np.array(fpr_list).mean(), "precision":np.array(precision_list).mean()}
    else:
        return tpr_list, fnr_list, fpr_list, precision_list
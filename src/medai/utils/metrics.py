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

import torch
import numpy as np

from sklearn.metrics import (f1_score, 
                             precision_score, 
                             recall_score, 
                             roc_auc_score,
                             zero_one_loss)

from torchmetrics.classification import (BinaryAUROC, 
                                         BinaryAccuracy,
                                         BinaryF1Score, 
                                         BinaryPrecision, 
                                         BinaryRecall, 
                                         BinarySpecificity,
                                         BinaryAveragePrecision)


class BinaryMetric_:
        
    def F1Score(y_target, y_pred):
        return f1_score(y_target, y_pred, average="binary")
        
    def precision(y_target, y_pred):
        return precision_score(y_target, y_pred, average="binary")
    
    def recall(y_target, y_pred):
        return recall_score(y_target, y_pred, average="binary")
    
    def zero_one(y_target, y_pred):
        return zero_one_loss(y_target, y_pred)
    
    def auroc(y_target, y_pred):
        pass
    
class BinaryMetric:
    def accuracy(preds, targets, threshold=0.5):
        metric = BinaryAccuracy(threshold=threshold)
        return metric(preds, targets)
    
    def f1_score(preds, targets, threshold=0.5):
        metric = BinaryF1Score(threshold=threshold)
        return metric(preds, targets)
        
    def precision(preds, targets, threshold=0.5):
        metric = BinaryPrecision(threshold=threshold)
        return metric(preds, targets)
    
    def recall(preds, targets, threshold=0.5):
        metric = BinaryRecall(threshold=threshold)
        return metric(preds, targets)
    
    def auroc(preds, targets):
        metric = BinaryAUROC(thresholds=None)
        return metric(preds, targets)
    
    def specificity(preds, targets, threshold=0.5):
        metric = BinarySpecificity(threshold=threshold)
        return metric(preds, targets)
    
    def avg_precision(preds, targets):
        metric = BinaryAveragePrecision(threshold=None)
        return metric(preds, targets)
    

if __name__ == "__main__":
    
    y_target = [1,1,1,1,1,1,1,1]
    y_pred =   [1,0,1,0,0,0,0,1]
    
    f1_score_result = BinaryMetric_.F1Score(y_target, y_pred)
    precision_result = BinaryMetric_.precision(y_target, y_pred)
    recall_result = BinaryMetric_.recall(y_target, y_pred)
    zero_one_result = BinaryMetric_.zero_one(y_target, y_pred)
    
    print("{} {} {} {}".format(f1_score_result, precision_result, recall_result, zero_one_result))
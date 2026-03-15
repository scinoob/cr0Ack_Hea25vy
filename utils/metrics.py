"""
Evaluation Metrics for Crack Segmentation
评价指标：F1-score、mIoU、Recall、Precision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegmentationMetrics:
    """
    Segmentation metrics calculator
    
    Metrics:
    - F1-score: Harmonic mean of precision and recall
    - mIoU: Mean Intersection over Union
    - Recall: True Positive Rate
    - Precision: Positive Predictive Value
    """
    
    def __init__(self, threshold=0.5, smooth=1e-6):
        """
        Args:
            threshold: Threshold for binary prediction
            smooth: Smoothing factor to avoid division by zero
        """
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives
        self.intersection = 0
        self.union = 0
        self.total_samples = 0
    
    def update(self, pred, target):
        """
        Update metrics with a batch of predictions
        
        Args:
            pred: Predicted probability map (B, 1, H, W) or (B, H, W)
            target: Ground truth mask (B, 1, H, W) or (B, H, W)
        """
        # Ensure same shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Binarize prediction
        pred = (pred > self.threshold).float()
        target = (target > 0.5).float()
        
        # Calculate TP, FP, FN, TN
        tp = torch.sum(pred * target).item()
        fp = torch.sum(pred * (1 - target)).item()
        fn = torch.sum((1 - pred) * target).item()
        tn = torch.sum((1 - pred) * (1 - target)).item()
        
        # Calculate intersection and union for IoU
        intersection = torch.sum(pred * target).item()
        union = torch.sum((pred + target) > 0).item()
        
        # Accumulate
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.intersection += intersection
        self.union += union
        self.total_samples += pred.size(0)
    
    def get_precision(self):
        """
        Calculate Precision (Positive Predictive Value)
        Precision = TP / (TP + FP)
        """
        precision = self.tp / (self.tp + self.fp + self.smooth)
        return precision
    
    def get_recall(self):
        """
        Calculate Recall (True Positive Rate, Sensitivity)
        Recall = TP / (TP + FN)
        """
        recall = self.tp / (self.tp + self.fn + self.smooth)
        return recall
    
    def get_f1_score(self):
        """
        Calculate F1-score (Dice Coefficient)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
           = 2 * TP / (2 * TP + FP + FN)
        """
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn + self.smooth)
        return f1
    
    def get_miou(self):
        """
        Calculate mIoU (Mean Intersection over Union)
        IoU = Intersection / Union
        """
        miou = self.intersection / (self.union + self.smooth)
        return miou
    
    def get_iou(self):
        """Alias for get_miou"""
        return self.get_miou()
    
    def get_accuracy(self):
        """
        Calculate Pixel Accuracy
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.smooth)
        return accuracy
    
    def get_results(self):
        """Get all metrics as a dictionary"""
        return {
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'miou': self.get_miou(),
            'accuracy': self.get_accuracy()
        }
    
    def print_results(self, prefix=''):
        """Print all metrics"""
        results = self.get_results()
        print(f"{prefix}Precision: {results['precision']:.4f}")
        print(f"{prefix}Recall:    {results['recall']:.4f}")
        print(f"{prefix}F1-Score:  {results['f1_score']:.4f}")
        print(f"{prefix}mIoU:      {results['miou']:.4f}")
        print(f"{prefix}Accuracy:  {results['accuracy']:.4f}")


class DiceCoefficient(nn.Module):
    """
    Dice Coefficient (F1-score) as a loss module
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceCoefficient, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Dice coefficient
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return dice


class IoUScore(nn.Module):
    """
    IoU (Intersection over Union) as a module
    """
    
    def __init__(self, smooth=1e-6):
        super(IoUScore, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            IoU score
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = ((pred + target) > 0).float().sum()
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return iou


def calculate_metrics_batch(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate metrics for a batch
    
    Args:
        pred: Predicted probability map (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        threshold: Threshold for binary prediction
        smooth: Smoothing factor
    Returns:
        Dictionary of metrics
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # Binarize prediction
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Calculate TP, FP, FN, TN
    tp = (pred_binary * target_binary).sum(dim=(1, 2))
    fp = (pred_binary * (1 - target_binary)).sum(dim=(1, 2))
    fn = ((1 - pred_binary) * target_binary).sum(dim=(1, 2))
    tn = ((1 - pred_binary) * (1 - target_binary)).sum(dim=(1, 2))
    
    # Calculate metrics
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    
    # IoU
    intersection = (pred_binary * target_binary).sum(dim=(1, 2))
    union = ((pred_binary + target_binary) > 0).float().sum(dim=(1, 2))
    iou = intersection / (union + smooth)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1.mean().item(),
        'miou': iou.mean().item(),
        'accuracy': accuracy.mean().item()
    }


def get_metrics_table(metrics_dict, title='Metrics'):
    """
    Format metrics as a table string
    
    Args:
        metrics_dict: Dictionary of metrics
        title: Table title
    Returns:
        Formatted string
    """
    table = f"\n{'='*50}\n"
    table += f"{title:^50}\n"
    table += f"{'='*50}\n"
    table += f"{'Metric':<20} {'Value':>20}\n"
    table += f"{'-'*50}\n"
    
    for key, value in metrics_dict.items():
        table += f"{key.capitalize():<20} {value:>20.4f}\n"
    
    table += f"{'='*50}\n"
    
    return table

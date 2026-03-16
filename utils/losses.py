"""
Loss Functions for Crack Segmentation
损失函数：主损失(Dice + BCE)、边界损失、辅助损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    Formula: L_dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Dice loss
        """
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy Loss with logits
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred, target):
        return self.loss(pred, target)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss
    
    Formula: L = L_bce + L_dice
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Combined loss
        """
        # ============================================================================
        # 改动日期: 2026-03-14
        # 改动作者: @kimi
        # 改动说明: 添加 torch.clamp 确保预测值在 [0, 1] 范围内
        # 原因: 当 DSAM 中的 sigmoid 被注释掉时，模型输出可能超出有效范围
        # ============================================================================
        pred = torch.clamp(pred, 0, 1)
        # ============================================================================

        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss
    
    Supervises crack boundaries using Sobel edge detection on ground truth.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred_boundary, target):
        """
        Args:
            pred_boundary: Predicted boundary map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Boundary loss
        """
        # ============================================================================
        # 改动日期: 2026-03-14
        # 改动作者: @kimi
        # 改动说明: 添加 torch.clamp 确保预测值在 [0, 1] 范围内
        # 原因: 当 DSAM 中的 sigmoid 被注释掉时，模型输出可能超出有效范围
        # ============================================================================
        pred_boundary = torch.clamp(pred_boundary, 0, 1)
        # ============================================================================

        # Extract edges from target using Sobel operator
        target_boundary = self._sobel_edge(target)

        # BCE loss
        loss = self.bce(pred_boundary, target_boundary)

        return loss

    def _sobel_edge(self, x):
        """
        Apply Sobel operator to extract edges
        
        Args:
            x: Input tensor (B, 1, H, W)
        Returns:
            Edge map (B, 1, H, W)
        """
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        # Apply convolution
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)

        # Combine edges
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        # Normalize to [0, 1]
        edge = torch.clamp(edge, 0, 1)

        return edge


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Formula: FL = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Focal loss
        """
        bce_loss = self.bce(pred, target)

        pt = torch.where(target == 1, pred, 1 - pred)

        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance
    
    Formula: Tversky = TP / (TP + α*FN + β*FP)
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability map (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            Tversky loss
        """
        pred = pred.view(-1)
        target = target.view(-1)

        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        fp = (pred * (1 - target)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        return 1 - tversky


# 使用TverskyLoss
class CombinedLoss(nn.Module):
    def __init__(self, lambda_boundary=0.0, lambda_aux=0.4,
                 bce_weight=1, dice_weight=1):
        super().__init__()

        self.lambda_boundary = lambda_boundary
        self.lambda_aux = lambda_aux

        # 更换损失函数为TverskyLoss
        # Main loss
        self.main_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

        # Boundary loss
        self.boundary_loss = BoundaryLoss()

        # Auxiliary loss
        self.aux_loss = TverskyLoss(alpha=0.8, beta=0.2, smooth=1)

    def forward(self, pred_main, pred_boundary, pred_aux, target):
        # Main loss
        loss_main = self.main_loss(pred_main, target)

        pred_boundary_up = F.interpolate(
            pred_boundary,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        # 直接使用原尺寸 target 计算高清边缘 Loss
        loss_boundary = self.boundary_loss(pred_boundary_up, target)

        pred_aux_up = F.interpolate(
            pred_aux,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        loss_aux = self.aux_loss(pred_aux_up, target)

        # Total loss
        total_loss = loss_main + self.lambda_boundary * loss_boundary + self.lambda_aux * loss_aux

        # Loss dictionary
        loss_dict = {
            'total': total_loss.item(),
            'main': loss_main.item(),
            'boundary': loss_boundary.item(),
            'aux': loss_aux.item()
        }

        return total_loss, loss_dict


# 原本的组合损失
class ThreeCombinedLoss(nn.Module):
    """
    Combined Loss for Crack Segmentation
    
    Formula: L_total = L_main + λ1 * L_boundary + λ2 * L_aux
    
    Where:
    - L_main: BCEDiceLoss (main segmentation loss)
    - L_boundary: BoundaryLoss (boundary supervision)
    - L_aux: BCEDiceLoss (auxiliary deep supervision)
    """

    def __init__(self, lambda_boundary=0.5, lambda_aux=0.4,
                 bce_weight=1.0, dice_weight=1.0):
        super().__init__()

        self.lambda_boundary = lambda_boundary
        self.lambda_aux = lambda_aux

        # Main loss
        self.main_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

        # Boundary loss
        self.boundary_loss = BoundaryLoss()

        # Auxiliary loss
        self.aux_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    def forward(self, pred_main, pred_boundary, pred_aux, target):
        """
        Args:
            pred_main: Main prediction (B, 1, H, W)
            pred_boundary: Boundary prediction (B, 1, H/4, W/4)
            pred_aux: Auxiliary prediction (B, 1, H/32, W/32)
            target: Ground truth mask (B, 1, H, W)
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Main loss
        loss_main = self.main_loss(pred_main, target)

        # Boundary loss (downsample target to match boundary prediction)
        # target_boundary = F.interpolate(target, size=pred_boundary.shape[2:],
        #                                 mode='bilinear', align_corners=False)
        # 不要下采样 target，而是上采样 pred_boundary 到原图尺寸！
        pred_boundary_up = F.interpolate(
            pred_boundary,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        # 直接使用原尺寸 target 计算高清边缘 Loss
        loss_boundary = self.boundary_loss(pred_boundary_up, target)

        # Auxiliary loss (downsample target to match auxiliary prediction)
        # target_aux = F.interpolate(target, size=pred_aux.shape[2:],
        #                            mode='bilinear', align_corners=False)
        # Auxiliary loss 逻辑也可以同步优化，虽然影响不大，但上采样预测比下采样 GT 更精准
        pred_aux_up = F.interpolate(
            pred_aux,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        loss_aux = self.aux_loss(pred_aux_up, target)

        # Total loss
        total_loss = loss_main + self.lambda_boundary * loss_boundary + self.lambda_aux * loss_aux

        # Loss dictionary
        loss_dict = {
            'total': total_loss.item(),
            'main': loss_main.item(),
            'boundary': loss_boundary.item(),
            'aux': loss_aux.item()
        }

        return total_loss, loss_dict


def build_criterion(config):
    """
    Build loss function from config
    
    Args:
        config: Training configuration
    Returns:
        criterion: Loss function
    """
    criterion = CombinedLoss(
        lambda_boundary=config.lambda_boundary,
        lambda_aux=config.lambda_aux,
        bce_weight=1.0,
        dice_weight=1.0
    )
    return criterion

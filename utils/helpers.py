"""
Helper Functions for Training and Evaluation
辅助函数：学习率调度、检查点保存/加载、日志记录、梯度可视化等
"""
import os
import torch
import numpy as np
import random
from datetime import datetime
import logging
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_current_time():
    """Get current time as formatted string"""
    return datetime.now().strftime('%Y-%m-%d_%H')


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
    Returns:
        logger: Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self, name=''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class LRScheduler:
    """
    Learning Rate Scheduler
    
    Supports:
    - Cosine annealing
    - Step decay
    - Warmup
    """
    
    def __init__(self, optimizer, scheduler_type='cosine', 
                 warmup_epochs=5, total_epochs=200, 
                 base_lr=1e-4, min_lr=1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'step')
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            base_lr: Base learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """
        Update learning rate
        
        Args:
            epoch: Current epoch (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        """Get current learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            return self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Main training phase
            if self.scheduler_type == 'cosine':
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            elif self.scheduler_type == 'step':
                # Decay by 0.1 every 50 epochs
                decay_epochs = [50, 100, 150]
                lr = self.base_lr
                for decay_epoch in decay_epochs:
                    if self.current_epoch >= decay_epoch:
                        lr *= 0.1
                return max(lr, self.min_lr)
            else:
                return self.base_lr


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth', is_best=False):
    """
    Save checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on
    Returns:
        epoch: Epoch number from checkpoint
        best_metric: Best metric from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    return epoch, best_metric


def get_optimizer(model, config):
    """
    Get optimizer
    
    Args:
        model: Model to optimize
        config: Training configuration
    Returns:
        optimizer: PyTorch optimizer
    """
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    return optimizer


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds):
    """
    Format seconds to human-readable time
    
    Args:
        seconds: Time in seconds
    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """
    Early stopping to stop training when metric stops improving
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if should stop
        
        Args:
            score: Current metric score
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def visualize_predictions(images, masks, preds, save_path=None, num_samples=4):
    """
    Visualize predictions
    
    Args:
        images: Input images (B, 3, H, W)
        masks: Ground truth masks (B, 1, H, W)
        preds: Predicted masks (B, 1, H, W)
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    
    num_samples = min(num_samples, images.size(0))
    
    _, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        mask = masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = preds[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# note that wrapper functions are used for Python closure
# so that we can pass arguments.

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads.append((module_name, grad))
    return hook

def get_all_layers(model, hook_forward, hook_backward):
    """Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        - layers: a dict with keys as layer/module and values as layer/module names
                  e.g. layers[nn.Conv2d] = layer1.0.conv1
        - grads: a list of tuples with module name and tensor output gradient
                 e.g. grads[0] == (layer1.0.conv1, tensor.Torch(...))
    """
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

# ============================================================================
# 改动日期: 2026-03-14
# 改动作者: @kimi
# 改动说明: 注释掉残留的实验代码，该代码在模块导入时执行但 model_bn 未定义
# ============================================================================
# 旧的代码:
# # register hooks
# layers_bn, grads_bn = get_all_layers(model_bn, hook_forward, hook_backward)
# ============================================================================
# 注意: 以下代码保留供参考， hooks 功能现在由 train.py 中的 register_gradient_hooks 实现
# 如需使用 get_all_layers 函数，请确保传入有效的 model 参数
# layers_bn, grads_bn = get_all_layers(model_bn, hook_forward, hook_backward)
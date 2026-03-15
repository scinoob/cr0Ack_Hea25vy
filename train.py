"""
Training Script for Crack Segmentation Network
训练脚本：包含训练、验证循环，支持评价指标计算
"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# 改动日期: 2026-03-14
# 改动作者: @kimi
# 改动说明: 引入 datetime 模块用于生成日期时间格式的目录名
# ============================================================================
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, update_config_from_args
from models import build_model
from datasets import get_dataloader
from utils import (
    SegmentationMetrics, CombinedLoss, set_seed, setup_logger, 
    AverageMeter, LRScheduler, save_checkpoint, load_checkpoint,
    get_optimizer, count_parameters, format_time, get_metrics_table,get_current_time
)


# ============================================================================
# 改动日期: 2026-03-14
# 改动作者: @kimi
# 改动说明: 添加梯度钩子相关全局变量和注册函数
# ============================================================================
# 存储梯度统计信息的字典
gradient_stats = {}
gradient_hooks = []


def register_gradient_hooks(model):
    """
    为指定模块注册梯度钩子，用于收集梯度统计信息
    
    注册的模块包括:
    - CNN branch1-4: cnn_branch.stages[0-3].blocks
    - MiT Block1-4: mit_branch.stages[0-3].blocks
    - Decoder DSAM: decoder.decoder4-1.dsam
    
    改动日期: 2026-03-14
    改动作者: @kimi
    """
    global gradient_hooks
    
    # 清除旧的钩子
    for hook in gradient_hooks:
        hook.remove()
    gradient_hooks = []
    
    # 定义钩子回调函数
    def make_hook(name):
        def hook_fn(module, grad_input, grad_output):
            # grad_output是tuple，取第一个元素
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                grad = grad_output[0]
                if grad is not None:
                    # 计算梯度统计信息
                    grad_mean = grad.abs().mean().item()
                    grad_max = grad.abs().max().item()
                    grad_std = grad.std().item()
                    
                    if name not in gradient_stats:
                        gradient_stats[name] = {'mean': [], 'max': [], 'std': []}
                    
                    gradient_stats[name]['mean'].append(grad_mean)
                    gradient_stats[name]['max'].append(grad_max)
                    gradient_stats[name]['std'].append(grad_std)
        return hook_fn
    
    # 注册 CNN branch1-4 的梯度钩子 (监控 blocks 的输出梯度)
    for i in range(4):
        stage = model.cnn_branch.stages[i]
        # 在 CNNStage 的 blocks 上注册钩子
        name = f'CNN_Branch_{i+1}'
        handle = stage.blocks.register_full_backward_hook(make_hook(name))
        gradient_hooks.append(handle)
    
    # 注册 MiT Block1-4 的梯度钩子 (监控每个 stage 的 blocks)
    for i in range(4):
        stage = model.mit_branch.stages[i]
        # 在 MiTStage 的 blocks ModuleList 上注册钩子
        name = f'MiT_Block_{i+1}'
        handle = stage.blocks.register_full_backward_hook(make_hook(name))
        gradient_hooks.append(handle)
    
    # 注册 Decoder DSAM 的梯度钩子
    decoder_dsams = [
        ('Decoder_DSAM_4', model.decoder.decoder4.dsam),
        ('Decoder_DSAM_3', model.decoder.decoder3.dsam),
        ('Decoder_DSAM_2', model.decoder.decoder2.dsam),
        ('Decoder_DSAM_1', model.decoder.decoder1.dsam),
    ]
    for name, module in decoder_dsams:
        handle = module.register_full_backward_hook(make_hook(name))
        gradient_hooks.append(handle)
    
    return len(gradient_hooks)


def log_gradient_stats(writer, epoch):
    """
    将梯度统计信息写入 TensorBoard
    每隔10个epoch调用一次
    
    改动日期: 2026-03-14
    改动作者: @kimi
    """
    global gradient_stats
    
    if writer is None or len(gradient_stats) == 0:
        return
    
    # 对每个模块记录平均梯度统计
    for name, stats in gradient_stats.items():
        if len(stats['mean']) > 0:
            avg_mean = sum(stats['mean']) / len(stats['mean'])
            avg_max = sum(stats['max']) / len(stats['max'])
            avg_std = sum(stats['std']) / len(stats['std'])
            
            writer.add_scalar(f'Gradient/{name}/mean', avg_mean, epoch)
            writer.add_scalar(f'Gradient/{name}/max', avg_max, epoch)
            writer.add_scalar(f'Gradient/{name}/std', avg_std, epoch)
    
    # 清空统计信息，为下一个周期做准备
    gradient_stats = {}


# ============================================================================
# 旧的代码:
# def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer=None):
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer=None):
    """
    Train for one epoch
    
    Args:
        model: Network model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch
        logger: Logger
        writer: TensorBoard writer
    Returns:
        avg_loss: Average loss for the epoch
        metrics: Dictionary of evaluation metrics
    """
    model.train()
    
    # Meters
    loss_meter = AverageMeter('Loss')
    loss_main_meter = AverageMeter('Loss_Main')
    loss_boundary_meter = AverageMeter('Loss_Boundary')
    loss_aux_meter = AverageMeter('Loss_Aux')
    
    # Metrics
    metrics = SegmentationMetrics()
    
    start_time = time.time()
    
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        masks = masks.to(device)
        step = images.shape[0]
        
        # Forward pass
        pred_main, pred_boundary, pred_aux = model(images)
        
        # Calculate loss
        loss, loss_dict = criterion(pred_main, pred_boundary, pred_aux, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        loss_main_meter.update(loss_dict['main'], images.size(0))
        loss_boundary_meter.update(loss_dict['boundary'], images.size(0))
        loss_aux_meter.update(loss_dict['aux'], images.size(0))
        
        # Update metrics
        metrics.update(pred_main.detach(), masks.detach())
        
        # Log batch progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            logger.info(
                f'Epoch [{epoch}][{batch_idx+1}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'(Main: {loss_main_meter.avg:.4f}, '
                f'Boundary: {loss_boundary_meter.avg:.4f}, '
                f'Aux: {loss_aux_meter.avg:.4f})'
            )
            # ============================================================================
            # 改动日期: 2026-03-14
            # 改动作者: @kimi
            # 改动说明: 添加梯度钩子后，保留原有的tensorboard图片记录功能
            # ============================================================================
            # 添加图片到tensorboard
            if writer is not None:
                writer.add_image('train/image',images[1],epoch)
                writer.add_image('train/pred',pred_main[1],epoch)
                writer.add_image('train/gt',masks[1],epoch)
    
    # Get metrics results
    metrics_results = metrics.get_results()
    
    # Log epoch summary
    elapsed_time = time.time() - start_time
    logger.info(f'Epoch [{epoch}] Training Summary:')
    logger.info(f'  Time: {format_time(elapsed_time)}')
    logger.info(f'  Loss: {loss_meter.avg:.4f}')
    logger.info(get_metrics_table(metrics_results, title='Training Metrics'))
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Loss_Main', loss_main_meter.avg, epoch)
        writer.add_scalar('Train/Loss_Boundary', loss_boundary_meter.avg, epoch)
        writer.add_scalar('Train/Loss_Aux', loss_aux_meter.avg, epoch)
        writer.add_scalar('Train/Precision', metrics_results['precision'], epoch)
        writer.add_scalar('Train/Recall', metrics_results['recall'], epoch)
        writer.add_scalar('Train/F1_Score', metrics_results['f1_score'], epoch)
        writer.add_scalar('Train/mIoU', metrics_results['miou'], epoch)
    
    return loss_meter.avg, metrics_results


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, logger, writer=None):
    """
    Validate the model
    
    Args:
        model: Network model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch
        logger: Logger
        writer: TensorBoard writer
    Returns:
        avg_loss: Average loss for validation
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Meters
    loss_meter = AverageMeter('Loss')
    loss_main_meter = AverageMeter('Loss_Main')
    loss_boundary_meter = AverageMeter('Loss_Boundary')
    loss_aux_meter = AverageMeter('Loss_Aux')
    
    # Metrics
    metrics = SegmentationMetrics()
    
    start_time = time.time()
    
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        pred_main, pred_boundary, pred_aux = model(images)
        
        # Calculate loss
        loss, loss_dict = criterion(pred_main, pred_boundary, pred_aux, masks)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        loss_main_meter.update(loss_dict['main'], images.size(0))
        loss_boundary_meter.update(loss_dict['boundary'], images.size(0))
        loss_aux_meter.update(loss_dict['aux'], images.size(0))
        
        # Update metrics
        metrics.update(pred_main, masks)
    
    # Get metrics results
    metrics_results = metrics.get_results()
    
    # Log validation summary
    elapsed_time = time.time() - start_time
    logger.info(f'Epoch [{epoch}] Validation Summary:')
    logger.info(f'  Time: {format_time(elapsed_time)}')
    logger.info(f'  Loss: {loss_meter.avg:.4f}')
    logger.info(get_metrics_table(metrics_results, title='Validation Metrics'))
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Val/Loss_Main', loss_main_meter.avg, epoch)
        writer.add_scalar('Val/Loss_Boundary', loss_boundary_meter.avg, epoch)
        writer.add_scalar('Val/Loss_Aux', loss_aux_meter.avg, epoch)
        writer.add_scalar('Val/Precision', metrics_results['precision'], epoch)
        writer.add_scalar('Val/Recall', metrics_results['recall'], epoch)
        writer.add_scalar('Val/F1_Score', metrics_results['f1_score'], epoch)
        writer.add_scalar('Val/mIoU', metrics_results['miou'], epoch)
    
    return loss_meter.avg, metrics_results


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Crack Segmentation Network')
    parser.add_argument('--dataset_type', type=str, default='crack500', 
                        choices=['crack500', 'cfd', 'mcd'],
                        help='Dataset type')
    parser.add_argument('--data_root', type=str, default='./data/crack500',
                        help='Root directory of dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    # ============================================================================
    # 改动日期: 2026-03-14
    # 改动作者: @kimi
    # 改动说明: 注释旧的 checkpoint_dir 和 log_dir 参数定义，改为自动创建日期时间目录
    # 旧的代码:
    # parser.add_argument('--checkpoint_dir', type=str, default=f'./checkpoints/{dataset_type}',
    #                     help='Directory to save checkpoints')
    # parser.add_argument('--log_dir', type=str, default='./logs',
    #                     help='Directory to save logs')
    # ============================================================================
    parser.add_argument('--output_dir', type=str, default='./output_train',
                        help='Root directory to save all training outputs (checkpoints and logs)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    config = update_config_from_args(config, args)
    
    # ============================================================================
    # 改动日期: 2026-03-14
    # 改动作者: @kimi
    # 改动说明: 创建统一的日期时间目录结构
    # 新结构: output_train/年-月-日-时-分/
    #         ├── checkpoints/     (存放 checkpoint)
    #         └── logs/            (存放 train.log)
    #             └── log/         (存放 tensorboard 文件)
    # ============================================================================
    # 生成日期时间字符串 (格式: 年-月-日-时-分)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # 创建主输出目录: output_train/日期时间/
    output_base_dir = os.path.join(args.output_dir, f"{args.dataset_type}_{timestamp}")
    
    # 创建子目录
    checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')
    log_dir = os.path.join(output_base_dir, 'logs')
    tensorboard_dir = os.path.join(log_dir, 'log')
    
    # 创建所有目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 更新配置中的路径
    config.train.checkpoint_dir = checkpoint_dir
    config.train.log_dir = log_dir
    
    # 旧的代码:
    # # Set random seed
    # set_seed(config.seed)
    # 
    # # Setup logger
    # log_file = os.path.join(config.train.log_dir, f"train_{get_current_time}.log")
    # logger = setup_logger('CrackSegmentation', log_file)
    # ============================================================================
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logger
    log_file = os.path.join(config.train.log_dir, 'train.log')
    logger = setup_logger('CrackSegmentation', log_file)
    
    logger.info('='*60)
    logger.info('Crack Segmentation Network - Training')
    logger.info('='*60)
    
    # ============================================================================
    # 改动日期: 2026-03-14
    # 改动作者: @kimi
    # 改动说明: 添加输出目录信息打印
    # ============================================================================
    logger.info('Output Directory Structure:')
    logger.info(f'  Base Directory: {output_base_dir}')
    logger.info(f'  Checkpoints:    {checkpoint_dir}')
    logger.info(f'  Logs:           {log_dir}')
    logger.info(f'  TensorBoard:    {tensorboard_dir}')
    logger.info('='*60)
    
    # Device
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # Create dataloaders
    logger.info('Creating dataloaders...')
    train_loader = get_dataloader(config.data, split='train')
    val_loader = get_dataloader(config.data, split='val')
    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model
    logger.info('Creating model...')
    model = build_model(config.model)
    model = model.to(device)
    
    # ============================================================================
    # 改动日期: 2026-03-14
    # 改动作者: @kimi
    # 改动说明: 注册梯度钩子用于监控CNN branch、MiT Block和Decoder DSAM的梯度
    # ============================================================================
    num_hooks = register_gradient_hooks(model)
    logger.info(f'Registered {num_hooks} gradient hooks for monitoring')
    # ============================================================================
    # 旧的代码:
    # # Count parameters
    # total_params, trainable_params = count_parameters(model)
    # logger.info(f'Total parameters: {total_params:,}')
    # logger.info(f'Trainable parameters: {trainable_params:,}')
    # ============================================================================
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Create optimizer
    optimizer = get_optimizer(model, config.train)
    logger.info(f'Optimizer: {config.train.optimizer}, LR: {config.train.lr}')
    
    # Create scheduler
    scheduler = LRScheduler(
        optimizer,
        scheduler_type=config.train.scheduler,
        warmup_epochs=config.train.warmup_epochs,
        total_epochs=config.train.epochs,
        base_lr=config.train.lr,
        min_lr=config.train.min_lr
    )
    
    # Create criterion
    criterion = CombinedLoss(
        lambda_boundary=config.train.lambda_boundary,
        lambda_aux=config.train.lambda_aux
    )
    
    # ============================================================================
    # 改动日期: 2026-03-14
    # 改动作者: @kimi
    # 改动说明: 修改 TensorBoard 日志路径为 logs/log/ 子目录
    # 旧的代码:
    # # TensorBoard writer
    # writer = None
    # if config.train.use_tensorboard:
    #     writer = SummaryWriter(os.path.join(config.train.log_dir, 'log'))
    # ============================================================================
    # TensorBoard writer
    writer = None
    if config.train.use_tensorboard:
        writer = SummaryWriter(tensorboard_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    if config.train.resume:
        logger.info(f'Resuming from checkpoint: {config.train.resume}')
        start_epoch, best_f1 = load_checkpoint(config.train.resume, model, optimizer, device)
        start_epoch += 1
        logger.info(f'Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}')
    
    # Training loop
    logger.info('='*60)
    logger.info('Starting training...')
    logger.info('='*60)
    
    for epoch in range(start_epoch, config.train.epochs):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch+1}/{config.train.epochs}')
        logger.info(f'Learning rate: {scheduler.get_lr():.6f}')
        logger.info(f'{"="*60}')
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, logger, writer
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch+1, logger, writer
        )
        
        # Update learning rate
        scheduler.step(epoch+1)
        
        # ============================================================================
        # 改动日期: 2026-03-14
        # 改动作者: @kimi
        # 改动说明: 每隔10个epoch将梯度统计信息写入TensorBoard
        # ============================================================================
        if (epoch + 1) % 10 == 0:
            log_gradient_stats(writer, epoch + 1)
            logger.info(f'Gradient statistics logged to TensorBoard at epoch {epoch + 1}')
        # ============================================================================
        # 旧的代码:
        # # Save checkpoint
        # is_best = val_metrics['f1_score'] > best_f1
        # if is_best:
        #     best_f1 = val_metrics['f1_score']
        # 
        # if (epoch + 1) % config.train.save_interval == 0 or is_best:
        # ============================================================================
        
        # Save checkpoint
        is_best = val_metrics['f1_score'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1_score']
        
        if (epoch + 1) % config.train.save_interval == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_f1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            save_checkpoint(
                checkpoint,
                config.train.checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch+1}.pth',
                is_best=is_best
            )
        
        logger.info(f'\nBest F1-Score so far: {best_f1:.4f}')
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    logger.info('='*60)
    logger.info('Training completed!')
    logger.info(f'Best F1-Score: {best_f1:.4f}')
    logger.info('='*60)


if __name__ == '__main__':
    main()

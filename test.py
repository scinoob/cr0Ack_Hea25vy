"""
Testing Script for Crack Segmentation Network
测试脚本：加载训练好的模型，在测试集上评估性能
"""
import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, update_config_from_args
from models import build_model
from datasets import get_dataloader
from utils import (
    SegmentationMetrics, CombinedLoss, set_seed, setup_logger,
    load_checkpoint, format_time, get_metrics_table, visualize_predictions
)


@torch.no_grad()
def test(model, dataloader, criterion, device, logger, save_predictions=False, output_dir=None):
    """
    Test the model
    
    Args:
        model: Network model
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to test on
        logger: Logger
        save_predictions: Whether to save prediction images
        output_dir: Directory to save predictions
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Meters
    from utils import AverageMeter
    loss_meter = AverageMeter('Loss')
    loss_main_meter = AverageMeter('Loss_Main')
    loss_boundary_meter = AverageMeter('Loss_Boundary')
    loss_aux_meter = AverageMeter('Loss_Aux')
    
    # Metrics
    metrics = SegmentationMetrics()
    
    # Storage for visualization
    all_images = []
    all_masks = []
    all_preds = []
    
    start_time = time.time()
    
    for batch_idx, (images, masks, img_paths) in enumerate(dataloader):
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
        
        # Save predictions
        if save_predictions and output_dir:
            for i in range(images.size(0)):
                img_name = os.path.basename(img_paths[i])
                name, ext = os.path.splitext(img_name)
                
                # Save prediction
                pred = pred_main[i, 0].cpu().numpy()
                pred_img = (pred * 255).astype(np.uint8)
                pred_pil = Image.fromarray(pred_img)
                pred_path = os.path.join(output_dir, f'{name}_pred.png')
                pred_pil.save(pred_path)
                
                # Save binary prediction
                pred_binary = (pred > 0.5).astype(np.uint8) * 255
                pred_binary_pil = Image.fromarray(pred_binary)
                pred_binary_path = os.path.join(output_dir, f'{name}_pred_binary.png')
                pred_binary_pil.save(pred_binary_path)
        
        # Store for visualization (first batch only)
        if batch_idx == 0:
            all_images = images[:4]
            all_masks = masks[:4]
            all_preds = pred_main[:4]
        
        # Log progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            logger.info(f'Test Progress: [{batch_idx+1}/{len(dataloader)}]')
    
    # Get metrics results
    metrics_results = metrics.get_results()
    
    # Log test summary
    elapsed_time = time.time() - start_time
    logger.info('='*60)
    logger.info('Test Summary:')
    logger.info(f'  Time: {format_time(elapsed_time)}')
    logger.info(f'  Loss: {loss_meter.avg:.4f}')
    logger.info(f'  Loss (Main): {loss_main_meter.avg:.4f}')
    logger.info(f'  Loss (Boundary): {loss_boundary_meter.avg:.4f}')
    logger.info(f'  Loss (Aux): {loss_aux_meter.avg:.4f}')
    logger.info(get_metrics_table(metrics_results, title='Test Metrics'))
    logger.info('='*60)
    
    # Visualize predictions
    if len(all_images) > 0:
        vis_path = os.path.join(output_dir, 'visualization.png') if output_dir else None
        visualize_predictions(all_images, all_masks, all_preds, save_path=vis_path)
        if vis_path:
            logger.info(f'Visualization saved to: {vis_path}')
    
    return metrics_results


@torch.no_grad()
def test_single_image(model, image_path, device, input_size=512):
    """
    Test on a single image
    
    Args:
        model: Network model
        image_path: Path to input image
        device: Device to test on
        input_size: Input image size
    Returns:
        prediction: Predicted mask
    """
    from torchvision import transforms
    
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    pred_main, _, _ = model(image_tensor)
    
    # Resize to original size
    pred = torch.nn.functional.interpolate(
        pred_main, size=(original_size[1], original_size[0]),
        mode='bilinear', align_corners=False
    )
    
    pred = pred[0, 0].cpu().numpy()
    
    return pred


def main():
    """Main testing function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Crack Segmentation Network')
    parser.add_argument('--dataset_type', type=str, default='crack500',
                        choices=['crack500', 'cfd', 'mcd'],
                        help='Dataset type')
    parser.add_argument('--data_root', type=str, default='./data/crack500',
                        help='Root directory of dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save predictions')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction images')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to test on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--single_image', type=str, default='',
                        help='Path to single image for inference')
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    config = update_config_from_args(config, args)
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logger
    log_file = os.path.join(args.output_dir, f'test_{get_current_time()}.log')
    logger = setup_logger('CrackSegmentation', log_file)
    
    logger.info('='*60)
    logger.info('Crack Segmentation Network - Testing')
    logger.info('='*60)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # Create model
    logger.info('Creating model...')
    model = build_model(config.model)
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.checkpoint}')
    epoch, best_metric = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f'Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.4f}')
    
    # Single image inference
    if args.single_image:
        logger.info(f'Running inference on single image: {args.single_image}')
        
        pred = test_single_image(model, args.single_image, device, config.data.input_size)
        
        # Save prediction
        os.makedirs(args.output_dir, exist_ok=True)
        img_name = os.path.basename(args.single_image)
        name, _ = os.path.splitext(img_name)
        
        # Save probability map
        pred_img = (pred * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img)
        pred_path = os.path.join(args.output_dir, f'{name}_pred.png')
        pred_pil.save(pred_path)
        logger.info(f'Prediction saved to: {pred_path}')
        
        # Save binary prediction
        pred_binary = (pred > 0.5).astype(np.uint8) * 255
        pred_binary_pil = Image.fromarray(pred_binary)
        pred_binary_path = os.path.join(args.output_dir, f'{name}_pred_binary.png')
        pred_binary_pil.save(pred_binary_path)
        logger.info(f'Binary prediction saved to: {pred_binary_path}')
        
        # Visualize
        if args.visualize:
            _, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            img = Image.open(args.single_image).convert('RGB')
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Probability map
            axes[1].imshow(pred, cmap='jet')
            axes[1].set_title('Probability Map')
            axes[1].axis('off')
            
            # Binary prediction
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title('Binary Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            vis_path = os.path.join(args.output_dir, f'{name}_visualization.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f'Visualization saved to: {vis_path}')
        
        return
    
    # Create dataloader
    logger.info('Creating dataloader...')
    test_loader = get_dataloader(config.data, split='test')
    logger.info(f'Test samples: {len(test_loader.dataset)}')
    
    # Create output directory
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f'Predictions will be saved to: {args.output_dir}')
    
    # Create criterion
    criterion = CombinedLoss(
        lambda_boundary=config.train.lambda_boundary,
        lambda_aux=config.train.lambda_aux
    )
    
    # Test
    logger.info('='*60)
    logger.info('Starting testing...')
    logger.info('='*60)
    
    metrics = test(
        model, test_loader, criterion, device, logger,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    logger.info('='*60)
    logger.info('Testing completed!')
    logger.info('='*60)


def get_current_time():
    """Get current time as formatted string"""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


if __name__ == '__main__':
    main()

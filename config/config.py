"""
Configuration file for Crack Segmentation Network
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Dataset type: 'crack500', 'cfd', 'mcd'
    dataset_type: str = 'cfd'

    # Data paths
    data_root: str = '/mnt/d/dev/data'
    train_images: str = 'train/image'
    train_masks: str = 'train/mask'
    val_images: str = 'validation/image'
    val_masks: str = 'validation/mask'
    test_images: str = 'test/image'
    test_masks: str = 'test/mask'

    # For MCD dataset
    train_list: str = 'train_slices.txt'
    val_list: str = 'val_slices.txt'
    test_list: str = 'test_slices.txt'

    # Image size
    input_size: int = 512

    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation: bool = True
    brightness_jitter: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Stem
    stem_in_channels: int = 3
    stem_out_channels: int = 64

    # MiT Branch
    mit_embed_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    mit_num_heads: List[int] = field(default_factory=lambda: [1, 2, 4])
    mit_sr_ratios: List[int] = field(default_factory=lambda: [8, 4, 2])
    mit_mlp_ratios: List[int] = field(default_factory=lambda: [4, 4, 4])
    mit_depths: List[int] = field(default_factory=lambda: [2, 2, 2])
    mit_drop_rate: float = 0.0
    mit_drop_path_rate: float = 0.1

    # CNN Branch
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])

    # DSAM
    dsam_kernel_size: int = 7

    # LEDIM
    ledim_kernel_size: int = 7

    # Decoder (3 stages: 128->64->32)
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32])


@dataclass
class TrainConfig:
    """Training configuration"""
    # Training
    epochs: int = 200
    batch_size: int = 16
    num_workers: int = 0

    # Optimizer
    optimizer: str = 'AdamW'
    # lr: float = 5e-4
    lr: float = 0.001
    betas = (0.9,0.999)
    eps:float = 1e-8
    weight_decay: float = 1e-2
    # weight_decay: float = 1e-4

    # Scheduler
    scheduler: str = 'cosine'
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    scheduler = 'CosineAnnealingLR'
    T_max:int = 50
    eta_min:float = 1e-5
    last_epoch:int = -1

    # Loss weights
    lambda_main: float = 1.0
    lambda_boundary: float = 0.5
    lambda_aux: float = 0.4

    # Checkpoint
    checkpoint_dir: str = './checkpoints'
    save_interval: int = 10

    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True
    log_dir: str = './logs'

    # Device
    device: str = 'cuda'

    # Resume training
    resume: str = ''


@dataclass
class TestConfig:
    """Testing configuration"""
    # Test
    batch_size: int = 1
    num_workers: int = 4

    # Checkpoint
    checkpoint_path: str = ''

    # Output
    save_predictions: bool = True
    output_dir: str = './results'

    # Visualization
    visualize: bool = True

    # Device
    device: str = 'cuda'


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    # Random seed
    seed: int = 42


def get_config():
    """Get default configuration"""
    return Config()


def update_config_from_args(config: Config, args):
    """Update config from command line arguments"""
    if hasattr(args, 'dataset_type') and args.dataset_type:
        config.data.dataset_type = args.dataset_type
    if hasattr(args, 'data_root') and args.data_root:
        config.data.data_root = args.data_root
    if hasattr(args, 'batch_size') and args.batch_size:
        config.train.batch_size = args.batch_size
        config.test.batch_size = args.batch_size
    if hasattr(args, 'epochs') and args.epochs:
        config.train.epochs = args.epochs
    if hasattr(args, 'lr') and args.lr:
        config.train.lr = args.lr
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir
    if hasattr(args, 'resume') and args.resume:
        config.train.resume = args.resume
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        config.test.checkpoint_path = args.checkpoint_path
    if hasattr(args, 'output_dir') and args.output_dir:
        config.test.output_dir = args.output_dir
    if hasattr(args, 'device') and args.device:
        config.train.device = args.device
        config.test.device = args.device
    if hasattr(args, 'seed') and args.seed is not None:
        config.seed = args.seed

    return config

"""
Crack Segmentation Dataset
支持三种数据集格式：crack500、CFD、MCD
"""
import os
import sys

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class CrackDataset(Dataset):
    """
    Crack Segmentation Dataset
    
    Supports three dataset formats:
    1. crack500: Standard format with train/test/validation folders
    2. cfd: CFD dataset format with specific naming conventions
    3. mcd: MCD dataset format with file lists
    """

    def __init__(self,
                 data_root,
                 dataset_type='crack500',
                 split='train',
                 input_size=512,
                 use_augmentation=True,
                 horizontal_flip=True,
                 vertical_flip=True,
                 rotation=True,
                 brightness_jitter=0.1,
                 use_clahe=False  # 新增参数控制是否使用 CLAHE
                 ):
        """
        Args:
            data_root: Root directory of the dataset
            dataset_type: Type of dataset ('crack500', 'cfd', 'mcd')
            split: Data split ('train', 'val', 'test')
            input_size: Input image size
            use_augmentation: Whether to use data augmentation
            horizontal_flip: Whether to apply horizontal flip
            vertical_flip: Whether to apply vertical flip
            rotation: Whether to apply rotation
            brightness_jitter: Brightness jitter factor
        """
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.split = split
        self.input_size = input_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.brightness_jitter = brightness_jitter
        self.use_clahe = use_clahe and (split == 'train')  # 建议仅在训练时使用，或者全部使用
        # 初始化 CLAHE 算法实例
        self.clahe = None
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Get image and mask paths
        self.image_paths, self.mask_paths = self._get_paths()

        # Basic transforms (resize and normalize)
        self.image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def _get_paths(self):
        """Get image and mask paths based on dataset type"""
        image_paths = []
        mask_paths = []

        if self.dataset_type == 'crack500':
            # crack500 format
            if self.split == 'train':
                image_dir = os.path.join(self.data_root, 'train', 'image')
                mask_dir = os.path.join(self.data_root, 'train', 'mask')
            elif self.split == 'val':
                image_dir = os.path.join(self.data_root, 'validation', 'image')
                mask_dir = os.path.join(self.data_root, 'validation', 'mask')
            else:  # test
                image_dir = os.path.join(self.data_root, 'test', 'image')
                mask_dir = os.path.join(self.data_root, 'test', 'mask')

            # Get all image files
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)
                    mask_path = os.path.join(mask_dir, img_name)
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

        elif self.dataset_type == 'cfd':
            # CFD format
            if self.split == 'train':
                image_dir = os.path.join(self.data_root, 'train', 'image')
                mask_dir = os.path.join(self.data_root, 'train', 'groundtruth')
            elif self.split == 'val':
                image_dir = os.path.join(self.data_root, 'validation', 'image')
                mask_dir = os.path.join(self.data_root, 'validation', 'groundtruth')
            else:  # test
                image_dir = os.path.join(self.data_root, 'test', 'image')
                mask_dir = os.path.join(self.data_root, 'test', 'groundtruth')

            # Get all image files
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)

                    # CFD has specific naming conventions
                    if self.split == 'val':
                        # Validation: image has _label suffix, mask doesn't
                        base_name = img_name.replace('_label.jpg', '').replace('_label.png', '').replace('.jpg', '')
                        mask_name = base_name + '_label.PNG'
                    else:
                        # Train and test: mask has _label suffix
                        base_name = img_name.replace('.jpg', '').replace('.png', '')
                        # base_name = img_name.replace('.jpg', '')
                        mask_name = base_name + '_label.PNG'

                    mask_path = os.path.join(mask_dir, mask_name)
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

        elif self.dataset_type == 'mcd':
            # MCD format with file lists
            image_dir = os.path.join(self.data_root, 'images')
            mask_dir = os.path.join(self.data_root, 'groundtruth')

            if self.split == 'train':
                list_file = os.path.join(self.data_root, 'train_slices.txt')
            elif self.split == 'val':
                list_file = os.path.join(self.data_root, 'val_slices.txt')
            else:  # test
                list_file = os.path.join(self.data_root, 'test_slices.txt')

            # Read file list
            with open(list_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]

            for file_name in file_names:
                img_path = os.path.join(image_dir, file_name + '.jpg')
                mask_path = os.path.join(mask_dir, file_name + '.png')
                image_paths.append(img_path)
                mask_paths.append(mask_path)

        return image_paths, mask_paths

    def _augment(self, image, mask):
        """Apply data augmentation"""
        # Random horizontal flip
        if self.horizontal_flip and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if self.vertical_flip and torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if self.rotation and torch.rand(1) > 0.5:
            angle = torch.randint(-2, 2, (1,)).item() * 90
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random brightness jitter
        if self.brightness_jitter > 0 and torch.rand(1) > 0.5:
            brightness_factor = 1.0 + torch.empty(1).uniform_(-self.brightness_jitter, self.brightness_jitter).item()
            image = TF.adjust_brightness(image, brightness_factor)

        return image, mask

    # 新增 CLAHE 处理函数
    def _apply_clahe(self, image_pil):
        # 转换 PIL 到 OpenCV (RGB)
        img_cv = np.array(image_pil)
        # 转换到 LAB 颜色空间分离亮度通道
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 仅对 L (亮度) 通道应用局部直方图均衡化
        cl = self.clahe.apply(l)

        # 合并通道并转回 RGB
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_clahe)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, H, W)
            mask: Tensor (1, H, W)
            image_path: str
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 新增：在进行ToTensor等张量转换前，应用 CLAHE
        if self.use_clahe:
            image = self._apply_clahe(image)

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Apply augmentation
        if self.use_augmentation:
            image, mask = self._augment(image, mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        return image, mask, img_path


def get_dataloader(config, split='train'):
    """
    Get dataloader for a specific split
    
    Args:
        config: Data configuration
        split: Data split ('train', 'val', 'test')
    Returns:
        dataloader: PyTorch DataLoader
    """

    dataset = CrackDataset(
        data_root=config.data_root,
        dataset_type=config.dataset_type,
        split=split,
        input_size=config.input_size,
        use_augmentation=config.use_augmentation,
        horizontal_flip=config.horizontal_flip,
        vertical_flip=config.vertical_flip,
        rotation=config.rotation,
        brightness_jitter=config.brightness_jitter
    )

    batch_size = config.batch_size if hasattr(config, 'batch_size') else 8
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 4
    shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


if __name__ == "__main__":
    # img = Image.open("D:/dev/data/CFD/train/image/046.jpg")
    batch_size = 1
    dataset = CrackDataset(
        data_root="D:/dev/data/CFD",
        dataset_type='cfd',
        split='train',
        input_size=512,
        use_augmentation=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        brightness_jitter=0.1
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    single = next(iter(dataloader))
    img, mask, meta = single
    print(img)
    from matplotlib import pyplot as plt

    plt.imshow(img.squeeze(0).numpy().transpose(1, 2, 0))
    plt.show()

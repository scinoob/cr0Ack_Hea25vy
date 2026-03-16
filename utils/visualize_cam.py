# ================= utils/visualize_cam.py =================
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 确保能导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model
from config import get_config


class SemanticSegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.eval()
        self.model.zero_grad()

        # 前向与反向传播
        pred_main, _, _ = self.model(x)
        target = pred_main.sum()
        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if len(gradients.shape) == 4:
            # CNN 层：标准 4D (B, C, H, W)
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.cpu().numpy()[0, 0]

        elif len(gradients.shape) == 3:
            # Transformer 层：3D 序列
            B, dim1, dim2 = gradients.shape

            # 智能判断空间维度 N 和通道维度 C
            # 因为输入是正方形，下采样后的空间序列长度 N 必定是完全平方数
            def is_perfect_square(val):
                return int(np.sqrt(val)) ** 2 == val

            if is_perfect_square(dim1) and not is_perfect_square(dim2):
                N, C = dim1, dim2
                spatial_dim = 1
                channel_dim = 2
            elif is_perfect_square(dim2) and not is_perfect_square(dim1):
                N, C = dim2, dim1
                spatial_dim = 2
                channel_dim = 1
            else:
                # 若都是或都不是平方数，回退到 MiT 标准的 (B, N, C)
                N, C = dim1, dim2
                spatial_dim = 1
                channel_dim = 2

            weights = torch.mean(gradients, dim=spatial_dim, keepdim=True)
            cam = torch.sum(weights * activations, dim=channel_dim)
            cam = F.relu(cam)

            H = int(np.sqrt(N))
            W = N // H

            # 安全 reshape 逻辑
            cam = cam.view(-1, H, W)[0].cpu().numpy()

        else:
            raise ValueError(f"不支持的特征图维度数量: {len(gradients.shape)}")

        # 安全归一化
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/model.pth")
    parser.add_argument('--image', type=str, default="./images/test.jpg")
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = get_args()
    device = args.device
    config = get_config()

    # 1. 加载模型
    model = build_model(config.model).to(device)
    checkpoint_path = args.checkpoint  # 注意：请在此处填入您实际权重的路径

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"【警告】未找到权重文件: {checkpoint_path}")

    # 2. 准备图像
    img_path = args.image  # 注意：请在此处填入您的测试图路径
    original_img = Image.open(img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 3. 目标层配置
    target_layers = {
        'CNN_Stage_1': model.cnn_branch.stages[0].blocks[-1],
        'CNN_Stage_2': model.cnn_branch.stages[1].blocks[-1],
        'CNN_Stage_3': model.cnn_branch.stages[2].blocks[-1],
        'CNN_Stage_4': model.cnn_branch.stages[3].blocks[-1],

        'MiT_Stage_1': model.mit_branch.stages[0].blocks[-1],
        'MiT_Stage_2': model.mit_branch.stages[1].blocks[-1],
        'MiT_Stage_3': model.mit_branch.stages[2].blocks[-1],
        'MiT_Stage_4': model.mit_branch.stages[3].blocks[-1],

        'Decoder_DSAM_4': model.decoder.decoder4.dsam,
        'Decoder_DSAM_3': model.decoder.decoder3.dsam,
        'Decoder_DSAM_2': model.decoder.decoder2.dsam,
        'Decoder_DSAM_1': model.decoder.decoder1.dsam,
    }

    # 4. 可视化绘制
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    orig_np = np.array(original_img.resize((512, 512)))

    for i in range(3):
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title(f"Original Image (Row {i + 1})")
        axes[i, 0].axis('off')

    layer_names = list(target_layers.keys())
    for idx, name in enumerate(layer_names):
        row = idx // 4
        col = (idx % 4) + 1

        print(f"正在生成特征热力图: {name}...")
        layer = target_layers[name]
        cam_extractor = SemanticSegmentationGradCAM(model, layer)
        cam_map = cam_extractor(input_tensor)

        cam_map_resized = cv2.resize(cam_map, (512, 512))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)

        axes[row, col].imshow(overlay)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = 'full_pipeline_grad_cam.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 全链路热力图分析已生成！请查看: {save_path}")


if __name__ == '__main__':
    main()

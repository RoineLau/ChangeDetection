import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scSE Module
class scSEModule(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.5):
        super(scSEModule, self).__init__()

        if in_channels <= 0:
            raise ValueError(f"In channels must be greater than 0. Received: {in_channels}.")

        reduced_channels = max(in_channels // 4, 1)

        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        channel_attention = self.channel_se(x)
        x = x * channel_attention

        spatial_attention = self.spatial_se(x)
        x = x * spatial_attention

        x = self.dropout(x)

        return x


# PSPNet Model
class PSPNetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPNetModel, self).__init__()
        self.model = smp.PSPNet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        return self.model(x)


class PSPNetWithAttention(PSPNetModel):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(PSPNetWithAttention, self).__init__(in_channels, out_channels)

        self.attention = scSEModule(out_channels, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.model(x)
        x = self.attention(x)
        x = self.dropout(x)

        return x


# 计算评估指标
def calculate_metrics(all_labels, all_preds):
    all_labels = np.ravel(all_labels)
    all_preds = np.ravel(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        misclassification_rate = (fp + fn) / (tp + tn + fp + fn)
        miss_rate = fn / (tp + fn) if tp + fn != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        kappa = cohen_kappa_score(all_labels, all_preds)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        misclassification_rate = 0
        miss_rate = 0
        accuracy = 0
        kappa = 0

    return misclassification_rate, miss_rate, accuracy, kappa


# 数据预处理
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
])


class ChangeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image1_path, image2_path, label_path, transform=None):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img1 = cv2.imread(self.image1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(self.image2_path, cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_path, cv2.IMREAD_GRAYSCALE)

        img = np.concatenate([img1, img2], axis=2)
        label = (label == 255).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        return img, label


import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time

def run_pspnet(image1_path, image2_path, label_path, model, save_folder, progress_bar, progress_var):
    """运行PSPNet进行变化检测并保存结果，同时更新进度条"""
    dataset = ChangeDetectionDataset(image1_path, image2_path, label_path, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model.eval()
    all_labels = []
    all_preds = []

    # 保存结果的文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    total_batches = len(loader)  # 获取总批次数
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(loader), desc='Testing', total=total_batches):
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).squeeze(1)
            preds = (preds > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            # 保存预测结果为图像
            pred_image = preds.squeeze(0).cpu().numpy()  # 去除批次维度
            pred_image = (pred_image * 255).astype(np.uint8)  # 转换为 0-255 范围

            # 保存为 PNG 图像
            pred_image_path = os.path.join(save_folder, f'psp_pred_{i+1}.png')
            Image.fromarray(pred_image).save(pred_image_path)

            # 更新进度条
            progress_var.set(int((i + 1) / total_batches * 100))  # 计算当前进度并设置
            progress_bar.update_idletasks()  # 更新进度条

            # 控制进度更新速度，避免过快更新导致UI卡顿
            time.sleep(0.05)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    misclassification_rate, miss_rate, accuracy, kappa = calculate_metrics(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}, Misclassification Rate: {misclassification_rate:.4f}, '
          f'Miss Rate: {miss_rate:.4f}, Kappa: {kappa:.4f}')

    # 最终进度条设置为100%
    progress_var.set(100)
    progress_bar.update_idletasks()

    return misclassification_rate, miss_rate, accuracy, kappa


# 加载模型
def load_model(model_path):
    model = PSPNetWithAttention(in_channels=6, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

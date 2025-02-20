import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torch.utils.data import WeightedRandomSampler
from PIL import Image

# 配置 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

torch.cuda.empty_cache()

# 计算错分率、漏分率、总体精度和Kappa系数
def calculate_metrics(all_labels, all_preds):
    all_labels = np.ravel(all_labels)
    all_preds = np.ravel(all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
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

# 数据集定义
class ChangeDetectionDataset(Dataset):
    def __init__(self, image1_dir, image2_dir, labels_dir, transform=None):
        self.image1_paths = sorted([os.path.join(image1_dir, f) for f in os.listdir(image1_dir) if f.endswith(('.png'))])
        self.image2_paths = sorted([os.path.join(image2_dir, f) for f in os.listdir(image2_dir) if f.endswith(('.png'))])
        self.labels_paths = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(('.png'))])
        self.transform = transform

        self.num_non_change_pixels = 0
        self.num_change_pixels = 0
        self._calculate_class_counts()

    def _calculate_class_counts(self):
        for label_path in self.labels_paths:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Invalid label at path {label_path}.")
            label = (label == 255).astype(np.uint8)
            self.num_change_pixels += np.sum(label == 1)
            self.num_non_change_pixels += np.sum(label == 0)
    
    def __len__(self):
        return len(self.image1_paths)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.image1_paths[idx], cv2.IMREAD_COLOR)
        img2 = cv2.imread(self.image2_paths[idx], cv2.IMREAD_COLOR)
        label = cv2.imread(self.labels_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None or label is None:
            raise ValueError(f"Invalid image or label at index {idx}. Check file paths.")

        img = np.concatenate([img1, img2], axis=2)

        # 标签二值化
        label = (label == 255).astype(np.uint8) 

        if self.transform:
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        return img, label

class scSEModule(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.5): #dropout参数
        super(scSEModule, self).__init__()
        
        if in_channels <= 0:
            raise ValueError(f"In channels must be greater than 0. Received: {in_channels}.")
        
        # 使用较小的缩小因子
        reduced_channels = max(in_channels // 4, 1)

        # 通道注意力模块
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力模块
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Dropout层
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # 计算通道注意力
        channel_attention = self.channel_se(x)
        x = x * channel_attention
        
        # 计算空间注意力
        spatial_attention = self.spatial_se(x)
        x = x * spatial_attention
        
        # 加入dropout
        x = self.dropout(x)

        return x

# 使用smp库中的PSPNet模型
class PSPNetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPNetModel, self).__init__()
        # 使用ResNet50作为PSPNet的骨干网络
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
        
        # 加入Dropout
        x = self.dropout(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, verbose=True, path='best_model.pth'):
        self.patience = patience 
        self.delta = delta
        self.verbose = verbose 
        self.path = path 
        self.counter = 0 
        self.best_loss = None 
        self.early_stop = False 
        self.best_model_wts = None 

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0 
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
            model.load_state_dict(self.best_model_wts) 
        return self.early_stop  

# 创建数据集实例
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
])

train_dataset = ChangeDetectionDataset(image1_dir='./LEVIR/train/A', image2_dir='./LEVIR/train/B', labels_dir='./LEVIR/train/label', transform=transform)
val_dataset = ChangeDetectionDataset(image1_dir='./LEVIR/val/A', image2_dir='./LEVIR/val/B', labels_dir='./LEVIR/val/label', transform=transform)

# 权重平衡
num_change_pixels = train_dataset.num_change_pixels
num_non_change_pixels = train_dataset.num_non_change_pixels
total_pixels = num_change_pixels + num_non_change_pixels

weights = [total_pixels / num_non_change_pixels if label == 0 else total_pixels / num_change_pixels for label in [0, 1]]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 创建带有注意力机制的PSPNet模型实例
model = PSPNetWithAttention(in_channels=6, out_channels=1).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}', ncols=100)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}] Training Loss: {avg_loss:.4f}')
    
    # 保存训练损失
    train_losses.append(avg_loss)
    
    return avg_loss

# 验证函数
def validate(model, val_loader, criterion, epoch, best_val_loss, save_dir='val_results2'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # 创建保存结果图的目录
    os.makedirs(save_dir, exist_ok=True)

    # 使用 tqdm 来显示一个进度条，而不是每个 batch 一个进度条
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f'Validating Epoch {epoch+1}', ncols=100, total=len(val_loader))):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.float())
            running_loss += loss.item()

            # 使用sigmoid处理输出得到概率值
            preds = torch.sigmoid(outputs).squeeze(1)  # 输出的概率值
            preds = (preds > 0.5).float()  # 阈值为0.5，转换为0或1的标签

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
                
            # 使用原始图像路径来命名预测图像
            for j, pred in enumerate(preds):
                original_image_path = val_loader.dataset.image1_paths[batch_idx * len(preds) + j]
                base_filename = os.path.basename(original_image_path)
                filename_without_extension = os.path.splitext(base_filename)[0]

                pred_binary = (pred.cpu().numpy() == 1).astype(np.uint8) * 255

                pred_path = os.path.join(save_dir, f"{filename_without_extension}_pred.png")

                cv2.imwrite(pred_path, pred_binary)

    avg_loss = running_loss / len(val_loader)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # 计算并打印评价指标
    misclassification_rate, miss_rate, accuracy, kappa = calculate_metrics(all_labels, all_preds)
    print(f'Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}, Misclassification Rate: {misclassification_rate:.4f}, '
          f'Miss Rate: {miss_rate:.4f}, Kappa: {kappa:.4f}')
    
    # 计算准确率并保存到列表
    val_accuracy.append(accuracy)

    return avg_loss, avg_loss < best_val_loss

# 在训练和验证时，确保记录每个epoch的损失值
train_losses = []
val_losses = []
val_accuracy = []

# 训练和验证过程
num_epochs = 100
early_stopping = EarlyStopping(patience=15, delta=0.001, verbose=True)  # 设置早停策略
best_val_loss = float('inf')
epoch_count = 0  # 实际训练的 epoch 数量

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    val_loss, is_best = validate(model, val_loader, criterion, epoch, best_val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    epoch_count += 1

    if is_best:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_pspnet_model_3.pth')
        print("Best model saved")

    if early_stopping(val_loss, model):
        print(f"Stopping early at epoch {epoch+1}")
        break

    
# 载入最佳模型
model.load_state_dict(torch.load('best_pspnet_model_3.pth'))

# 可视化训练过程中的损失和精度
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_plot3.png")
plt.show()

# 绘制精度图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot3.png')
plt.show()

####### 测试部分 #######
def test(model, test_loader, save_dir='CD_test_results3'):
    model.eval()
    all_labels = []
    all_preds = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).squeeze(1)
            preds = (preds > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            for j, pred in enumerate(preds):
                pred_map = pred.squeeze(0).cpu().numpy()

                if pred_map.ndim == 3: 
                    pred_map = pred_map[:, :, 0] 

                pred_binary = (pred_map == 1).astype(np.uint8) * 255 
                
                original_image_path = test_loader.dataset.image1_paths[batch_idx * len(preds) + j]
                base_filename = os.path.basename(original_image_path)
                filename_without_extension = os.path.splitext(base_filename)[0]

                result_image_path = os.path.join(save_dir, f"{filename_without_extension}.png")
                
                cv2.imwrite(result_image_path, pred_binary)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    misclassification_rate, miss_rate, accuracy, kappa = calculate_metrics(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}, Misclassification Rate: {misclassification_rate:.4f}, '
          f'Miss Rate: {miss_rate:.4f}, Kappa: {kappa:.4f}')

# 创建测试集并进行测试
test_dataset = ChangeDetectionDataset(image1_dir='./LEVIR/test/A', image2_dir='./LEVIR/test/B', labels_dir='./LEVIR/test/label', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2)

test(model, test_loader)

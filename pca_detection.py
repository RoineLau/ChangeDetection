#PCA
import numpy as np
import os
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.morphology import closing, opening, disk
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 读取多波段影像
def read_multiband_image(image_path):
    """读取多波段遥感影像"""
    image = Image.open(image_path)
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # 单波段情况
        image_array = np.expand_dims(image_array, axis=-1)
    return image_array

# 调整影像大小
def resize_images(image1, image2):
    """调整两幅影像到相同尺寸"""
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1_resized = image1[:min_height, :min_width, :]
    image2_resized = image2[:min_height, :min_width, :]
    return image1_resized, image2_resized

# PCA分析
def apply_pca(image, n_components=3):
    """对影像进行PCA分析"""
    reshaped_image = image.reshape(-1, image.shape[2])  # 展平为二维
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(reshaped_image)
    return transformed.reshape(image.shape[0], image.shape[1], n_components)

# CVA计算差异影像
def calculate_cva(image1_pca, image2_pca):
    """计算变化矢量分析的差异影像"""
    diff_image = np.sqrt(np.sum((image1_pca - image2_pca) ** 2, axis=2))
    return diff_image

# 图像预处理：高斯滤波去噪
def preprocess_image(image, sigma=1.0):
    """图像预处理：高斯滤波去噪"""
    return gaussian_filter(image, sigma=sigma)

# 使用Otsu算法计算全局阈值
def otsu_threshold(diff_image):
    """使用Otsu算法自动计算全局阈值"""
    return threshold_otsu(diff_image)

# 使用Canny边缘检测突出变化区域
def edge_detection(diff_image):
    """使用Canny边缘检测突出变化区域"""
    edges = canny(diff_image)
    return edges.astype(np.uint8) * 255

# 使用局部阈值检测变化区域
def local_threshold(diff_image, window_size=50):
    """使用局部阈值检测变化区域"""
    h, w = diff_image.shape
    half_window = window_size // 2
    padded_diff = np.pad(diff_image, pad_width=half_window, mode='reflect')
    change_map = np.zeros_like(diff_image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            # 提取局部窗口
            window = padded_diff[i:i + window_size, j:j + window_size]
            local_threshold_value = np.mean(window) + np.std(window)  # 使用局部均值加标准差

            # 应用局部阈值
            change_map[i, j] = 255 if diff_image[i, j] > local_threshold_value else 0

    return change_map

# 结合全局阈值和局部阈值
def combined_threshold(diff_image):
    """结合全局阈值和局部阈值"""
    global_threshold = otsu_threshold(diff_image)
    local_map = local_threshold(diff_image, window_size=50)
    edge_map = edge_detection(diff_image)

    # 结合全局阈值与局部阈值检测
    combined_map = np.zeros_like(diff_image, dtype=np.uint8)
    combined_map[diff_image > global_threshold] = 255  # 全局阈值检测
    combined_map[local_map == 255] = 255  # 局部阈值检测
    combined_map[edge_map == 255] = 255  # 边缘检测检测

    return combined_map

# 后处理：应用形态学操作去除噪声
def post_process_change_map(change_map):
    """后处理：腐蚀去除小噪点，然后膨胀恢复边界"""
    # 增大腐蚀操作的结构元素，去除更多噪声
    eroded_map = opening(change_map, disk(5))
    # 增强膨胀操作，保持边界
    processed_map = closing(eroded_map, disk(5))
    return processed_map

# 保存变化检测图
def save_change_map(change_map, save_path):
    """保存变化检测结果到指定路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    change_map_image = Image.fromarray(change_map)
    change_map_image.save(save_path)
    #print(f"Change detection map saved at {save_path}")

# 计算评价指标：总体精度、错分率、漏分率、Kappa系数
def calculate_metrics(true_map, pred_map):
    true_map = (true_map > 0).astype(np.uint8)
    pred_map = (pred_map > 0).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(true_map.flatten(), pred_map.flatten()).ravel()

    overall_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    kappa = (overall_accuracy - ((fp + fn) * (fp + tp) + (fn + tn) * (tp + tn)) / (tp + tn + fp + fn) ** 2) / \
            (1 - ((fp + fn) * (fp + tp) + (fn + tn) * (tp + tn)) / (tp + tn + fp + fn) ** 2)

    miss_rate = fn / (fn + tp) if (fn + tp) != 0 else 0
    commission_error = fp / (fp + tn) if (fp + tn) != 0 else 0

    return overall_accuracy, kappa, miss_rate, commission_error

def process_images(image_folder1, image_folder2, gt_folder, save_folder):
    """对文件夹下多张图片进行变化检测"""
    all_metrics = []

    for file_name in os.listdir(image_folder1):
        image1_path = os.path.join(image_folder1, file_name)
        image2_path = os.path.join(image_folder2, file_name)

        image1 = read_multiband_image(image1_path)
        image2 = read_multiband_image(image2_path)

        image1_resized, image2_resized = resize_images(image1, image2)

        # PCA分析
        image1_pca = apply_pca(image1_resized)
        image2_pca = apply_pca(image2_resized)

        # CVA计算差异影像
        diff_image = calculate_cva(image1_pca, image2_pca)

        # 图像去噪
        diff_image = preprocess_image(diff_image, sigma=1.0)

        # 变化检测：使用组合的阈值方法（全局+局部+边缘）
        change_map = combined_threshold(diff_image)

        # 后处理：应用形态学操作去除噪声
        processed_change_map = post_process_change_map(change_map)

        # 保存变化检测图
        save_path = os.path.join(save_folder, file_name)
        save_change_map(processed_change_map, save_path)

        gt_path = os.path.join(gt_folder, file_name)
        gt_map = np.array(Image.open(gt_path))

        # 计算评价指标
        metrics = calculate_metrics(gt_map, processed_change_map)
        all_metrics.append(metrics)

    # 计算所有图像的平均指标
    avg_metrics = np.mean(all_metrics, axis=0)
    return avg_metrics

# 主程序
image_folder1 = 'LEVIR/test/matched/' 
image_folder2 = 'LEVIR/test/B/' 
gt_folder = 'LEVIR/test/label/' 
save_folder = 'LEVIR/test/PCA_label/' 

avg_metrics = process_images(image_folder1, image_folder2, gt_folder, save_folder)
print(f"Average Metrics: \nOverall Accuracy: {avg_metrics[0]:.4f}, False Positive Rate: {avg_metrics[1]:.4f}, False Negative Rate: {avg_metrics[2]:.4f}, Kappa Coefficient: {avg_metrics[3]:.4f}")

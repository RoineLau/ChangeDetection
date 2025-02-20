import numpy as np
import os
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from skimage.morphology import opening, closing, disk
from sklearn.metrics import confusion_matrix
from PIL import Image
import time
from tkinter import messagebox

# 读取RGB三波段影像
def read_rgb_image(image_path):
    """读取多波段遥感影像"""
    image_ = Image.open(image_path)
    image = np.array(image_)
    if len(image.shape) == 2:  # 单波段情况
        image = np.expand_dims(image, axis=-1)
    return image

# 规范化处理
def normalize(image):
    return (image - np.mean(image)) / np.std(image)

# 去除常数波段（方差为零的波段）
def remove_constant_bands(image):
    variances = np.var(image, axis=(0, 1))
    return image[:, :, variances > 1e-6]  # 方差大于1e-6的波段

# 正则化逆处理
def regularized_inverse(matrix, epsilon=1e-6):
    return np.linalg.inv(matrix + epsilon * np.eye(matrix.shape[0]))

# 计算典型相关分析（CCA）的典型变量U, V
def compute_canonical_correlations(image1, image2):
    cca = CCA(n_components=image1.shape[1])
    cca.fit(image1, image2)
    U, V = cca.transform(image1, image2)
    return U, V

# 计算MAD（U-V）
def compute_mad(U, V):
    return U - V

# 执行EM迭代，计算MAD和卡方统计量
def em_iteration(image1, image2, max_iter=100, tol=1e-4):
    assert image1.shape == image2.shape, "Two images must have the same shape."
    num_pixels, num_bands = image1.shape
    weights = np.ones(num_pixels)  # 初始权重为1

    for iteration in range(max_iter):
        # 加权均值
        weighted_mean1 = np.average(image1, axis=0, weights=weights)
        weighted_mean2 = np.average(image2, axis=0, weights=weights)

        # 去中心化
        centered1 = image1 - weighted_mean1
        centered2 = image2 - weighted_mean2

        # 加权协方差
        weighted_cov11 = (centered1.T * weights) @ centered1 / np.sum(weights)
        weighted_cov22 = (centered2.T * weights) @ centered2 / np.sum(weights)
        weighted_cov12 = (centered1.T * weights) @ centered2 / np.sum(weights)

        # 正则化协方差矩阵
        weighted_cov11_inv = regularized_inverse(weighted_cov11)
        weighted_cov22_inv = regularized_inverse(weighted_cov22)

        # 典型相关分析 (CCA)
        U, V = compute_canonical_correlations(centered1, centered2)

        # 计算 MAD 和卡方统计量
        mad = compute_mad(U, V)
        chi_squared = np.sum(mad**2, axis=1)

        # 更新权重
        new_weights = 1 - chi2.cdf(chi_squared, df=num_bands)

        # 检查收敛
        if np.linalg.norm(new_weights - weights) < tol:
            break

        weights = new_weights

    return mad, chi_squared

# 生成变化检测图
def generate_change_map(chi_squared, threshold, height, width):
    binary_map = (chi_squared > threshold).astype(np.uint8) * 255
    return binary_map.reshape(height, width)

# 保存变化检测图到指定路径
def save_change_map(change_map, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    change_map_image = Image.fromarray(change_map)
    change_map_image.save(save_path)

# 后处理：应用形态学操作去除噪声
def post_process_change_map(change_map):
    # 腐蚀操作去除小噪点
    eroded_map = opening(change_map, disk(3))
    # 膨胀操作恢复边界
    processed_map = closing(eroded_map, disk(3))
    return processed_map

# 计算评价指标
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


# 运行IRMAD变化检测
def irmad_detection(image1_path, image2_path, save_folder, progress_bar, progress_var):
    """运行IRMAD变化检测并返回变化检测图"""
    print(f"Processing {os.path.basename(image1_path)} and {os.path.basename(image2_path)}...")

    # 读取影像并规范化
    image1 = normalize(read_rgb_image(image1_path))
    image2 = normalize(read_rgb_image(image2_path))

    # 去除常数波段
    image1 = remove_constant_bands(image1)
    image2 = remove_constant_bands(image2)

    # 展平影像
    flattened_image1 = image1.reshape(-1, image1.shape[2])
    flattened_image2 = image2.reshape(-1, image2.shape[2])

    # 执行 EM 迭代
    mad, chi_squared = em_iteration(flattened_image1, flattened_image2)

    # 计算卡方统计量的阈值，使用百分位
    threshold = np.percentile(chi_squared, 85)

    # 获取原始影像的高度和宽度
    height, width = image1.shape[:2]

    # 生成变化检测图
    change_map = generate_change_map(chi_squared, threshold, height, width)

    # 后处理：去除噪声
    processed_change_map = post_process_change_map(change_map)

    # 保存变化检测图
    save_path = os.path.join(save_folder, os.path.basename(image1_path).replace('.tif', '_change_map.png'))
    save_change_map(processed_change_map, save_path)

    # 更新进度条
    for i in range(50, 101):  # 变化检测部分的进度更新
        progress_var.set(i)
        progress_bar.update_idletasks()
        time.sleep(0.01)

    return processed_change_map


# 计算IRMAD精度
def calculate_irmad_accuracy(true_map, pred_map, progress_bar, progress_var):
    """计算IRMAD变化检测的精度指标"""
    overall_accuracy, kappa, miss_rate, commission_error = calculate_metrics(true_map, pred_map)

    # 更新进度条
    for i in range(101):  # 计算精度时的进度更新
        progress_var.set(i)
        progress_bar.update_idletasks()
        time.sleep(0.01)

    return overall_accuracy, kappa, miss_rate, commission_error


# 主运行函数，拆分为两步：IRMAD变化检测和精度计算
def run(image1_path, image2_path, label_path, save_folder, progress_bar, progress_var):
    # 步骤1：IRMAD变化检测
    processed_change_map = irmad_detection(image1_path, image2_path, save_folder, progress_bar, progress_var)
    
    # 步骤2：计算精度
    true_map = np.array(Image.open(label_path).convert('L'))  # 读取真实标签图
    overall_accuracy, kappa, miss_rate, commission_error = calculate_irmad_accuracy(true_map, processed_change_map, progress_bar, progress_var)

    # 显示精度结果
    messagebox.showinfo("IRMAD结果", f"IRMAD变化检测完成，精度：\n总体精度：{overall_accuracy:.4f}\n错检率：{commission_error:.4f}\n漏检率：{miss_rate:.4f}\nKappa系数：{kappa:.4f}")

    return overall_accuracy, kappa, miss_rate, commission_error
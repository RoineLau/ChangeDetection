import numpy as np
import os
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from skimage.morphology import opening, closing, disk
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 读取RGB三波段影像
def read_rgb_image(image_path):
    with rasterio.open(image_path) as src:
        image = np.stack([src.read(i + 1) for i in range(3)], axis=-1) 
    return image

# 规范化处理
def normalize(image):
    return (image - np.mean(image)) / np.std(image)

# 去除常数波段（方差为零的波段）
def remove_constant_bands(image):
    variances = np.var(image, axis=(0, 1))
    return image[:, :, variances > 1e-6]

# 正则化逆处理
def regularized_inverse(matrix, epsilon=1e-6):
    return np.linalg.inv(matrix + epsilon * np.eye(matrix.shape[0]))

# 计算典型相关分析（CCA）的典型变量U,V
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

        # 典型相关分析(CCA)
        U, V = compute_canonical_correlations(centered1, centered2)

        # 计算MAD和卡方统计量
        mad = compute_mad(U, V)
        chi_squared = np.sum(mad**2, axis=1)

        # 更新权重
        new_weights = 1 - chi2.cdf(chi_squared, df=num_bands)

        # 检查收敛
        if np.linalg.norm(new_weights - weights) < tol:
            #print(f"Converged after {iteration + 1} iterations.")
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

# 可视化变化检测图
def plot_change_map(change_map):
    plt.figure(figsize=(8, 6))
    plt.imshow(change_map, cmap='gray')
    plt.title('Change Detection Map (Black and White)')
    plt.axis('off')
    plt.show()

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
    
    miss_rate = fn / (fn + tp) if (fn + tp) != 0 else 0  # 避免除零
    commission_error = fp / (fp + tn) if (fp + tn) != 0 else 0

    return overall_accuracy, kappa, miss_rate, commission_error

# 主函数
def main():
    image1_folder = 'LEVIR/test/matched/' 
    image2_folder = 'LEVIR/test/B/' 
    label_folder = 'LEVIR/test/label/' 
    save_folder = 'LEVIR/test/IRMAD_label/' 

    image1_files = sorted(os.listdir(image1_folder))
    image2_files = sorted(os.listdir(image2_folder))
    label_files = sorted(os.listdir(label_folder))

    overall_accuracy_list = []
    kappa_list = []
    miss_rate_list = []
    commission_error_list = []

    for i in range(len(image1_files)):
        image1_path = os.path.join(image1_folder, image1_files[i])
        image2_path = os.path.join(image2_folder, image2_files[i])
        label_path = os.path.join(label_folder, label_files[i])

        save_path = os.path.join(save_folder, image1_files[i].replace('.tif', '_change_map.png'))

        # 读取影像并规范化
        print(f"Processing {image1_files[i]}...")
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
        save_change_map(processed_change_map, save_path)

        # 读取真实标签并计算指标
        true_map = np.array(Image.open(label_path).convert('L'))
        pred_map = processed_change_map

        overall_accuracy, kappa, miss_rate, commission_error = calculate_metrics(true_map, pred_map)
        overall_accuracy_list.append(overall_accuracy)
        kappa_list.append(kappa)
        miss_rate_list.append(miss_rate)
        commission_error_list.append(commission_error)

    # 计算平均指标
    print(f"\nAverage Overall Accuracy: {np.mean(overall_accuracy_list):.4f}")
    print(f"Average Kappa Coefficient: {np.mean(kappa_list):.4f}")
    print(f"Average Miss Rate: {np.mean(miss_rate_list):.4f}")
    print(f"Average Commission Error: {np.mean(commission_error_list):.4f}")

if __name__ == "__main__":
    main()

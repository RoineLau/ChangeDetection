# ChangeDetection

基于双时相遥感影像的三种变化监测方法，包括 **PCA法**、**IR-MAD法** 与 **融合scSE注意力机制的PSPNet法**。

## 技术路线
![image](https://github.com/user-attachments/assets/21d8e90f-a568-46fb-9615-4c60b3eca264)


## 1. 算法原理

### 1.1 主成分分析法（PCA）

**主成分分析法（Principal Component Analysis，简称PCA）** 是一种常用的数据降维技术。其原理主要是通过正交变换，将可能存在相关性的变量转换为一组线性不相关的变量，这些变量被称为主成分。

传统 PCA 法通过差分图来分析影像间的光谱特征差异，提取最能代表图像光谱属性的主成分分量来进行变化检测，从而构造差异影像。

**特点：**
- 能有效降低数据集的维数，实现数据压缩的同时尽可能多地保留遥感影像中的有用信息。
- 只对 PCA 变换后的一个主成分进行构造差异影像，可能导致变化信息缺失。

**改进方法：**
受文献 [1] 启发，本文尝试基于 **PCA 的 CVA 法**，具体步骤如下：
1. 对影像进行 PCA 变换，得到相关性较小的多个主分量。
2. 选取前 **3 个主分量** 进行 **CVA（Change Vector Analysis）** 分析，以获取信息充足且噪声较少的差异影像。
3. 构建差异影像后，对影像进行二值化显示。

其原理如公式（1）所示：
![image](https://github.com/user-attachments/assets/ff9bf2d9-139b-47d5-a4c6-a25c88f3e70a)

（式中，Ｘ代表各光谱变化差异图；Ｘ′表示阈值分割后的二值图；T是需要设置的阈值参数。）

### 阈值选择方法

1. **Otus法（Otsu's method）**
   - 自适应阈值选择方法，基于最大类间方差来确定最佳阈值。
   - 适用于双峰分布的图像，常用于图像分割和二值化处理。

2. **快速局部阈值法**
   - 对影像进行分块处理，计算均值和标准差，并根据这些统计量确定局部阈值。
   - 适用于光照不均或存在噪声的图像。

### 1.2 迭代加权多元变化检测算法（IR-MAD）

**核心原理：**
1. **典型相关分析（CCA）**
   - 通过计算典型变量对（U, V），最大化两时相影像的相关性。
2. **多变量差异（MAD）分量**
   - 计算 MAD 分量，有效消除影像间的辐射差异，仅保留可能由地物变化引起的差异。

3. **卡方统计量计算**
   - 计算 MAD 分量的加权平方和，描述变化程度。
4. **期望-最大化（EM）迭代**
   - 通过迭代计算重新评估 MAD 分量和卡方统计量，增强对实际地物变化的敏感性。

### 1.3 改进型 PSP-Net 建筑物变化检测算法

#### scSE 注意力机制模块

**scSE 模块** 结合了 **通道注意力（cSE）** 和 **空间注意力（sSE）**，增强 CNN 的特征学习能力。

#### PSP-Net 模型

PSP-Net（Pyramid Scene Parsing Network）核心在于 **金字塔池化模块（Pyramid Pooling Module）**，能够聚合不同区域的上下文信息，提高全局上下文特征的表达能力。

**特点：**
- 通过全局上下文信息的引导，准确判断建筑物是否发生变化。
- 强大的特征提取能力，适用于不同分辨率和复杂度的遥感影像。

---

## 📜 参考文献

[1]	黄维,黄进良,王立辉,等.基于PCA的变化向量分析法遥感影像变化检测[J].国土资源遥感,2016,28(01):22-27.

[2]	王晓雷,杨景鹏,江涛,等.IR-MAD算法的遥感影像变化检测方法研究[J].地理信息世界,2020,27(03):56-62.

[3]	Canty M y,Nielsen A A,Schmidt M.Automatic Radiometric Normalization of Multitemporal Satellite lmagery!.Remote Sensing of Environment,2003,91(3):441-451.

[4]	王盼盼,刘超,孙健飞,等.基于UNet模型的遥感影像建筑物变化检测研究[J].江西科学,2024,42(02):355-359.DOI:10.13990/j.issn1001-3679.2024.02.021.

[5]	Roy, A. G., Navab, N., & Wachinger, C. Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention;,2018:3.

[6]	刘海红,李延龙,李亚刚,等.基于改进型PSPNet模型的高分辨率遥感影像建筑物变化检测方法研究[J].经纬天地,2024,(02):1-4+22.

[7]	Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2881-2890.

---

📢 **以上内容节选自本人大三的遥感原理课程的期末大作业实验报告，内容可能有遗漏与错误之处，欢迎提出建议！** 🎯

import cv2
from skimage import segmentation
import numpy as np
from skimage.filters import gabor
from sklearn.cluster import KMeans
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.stats import norm
from sklearn.cluster import MeanShift
from scipy.ndimage import label

# devide the image into n segments
def segment(img): 
    # 將 BGR 圖像轉換為 LAB 色彩空間
    LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 使用 Quickshift 分割圖像
    seg = segmentation.quickshift(
        LAB_img,
        kernel_size=9,    # 對應 MATLAB 的 SpatialBandWidth
        max_dist=15,      # 對應 MATLAB 的 RangeBandWidth
        ratio=0.1         # 可調參數，用於控制區域間距離
    )
    # 獲取區域數量
    segnum = np.max(seg)  # 最大標籤值即為區域數量

    return seg, segnum
def segment_image(img, spatial_bandwidth=9, range_bandwidth=25, min_region_area=200):
    """
    使用 Mean Shift 分割图像并过滤小区域
    Args:
        img: 输入 BGR 图像 (numpy array, shape: [H, W, 3])
        spatial_bandwidth: 空间带宽，用于位置权重调整 (int)
        range_bandwidth: 颜色带宽，用于颜色权重调整 (int)
        min_region_area: 最小区域面积阈值 (int)
    Returns:
        seg: 分割后的标签矩阵 (numpy array, shape: [H, W])
        segnum: 分割区域数量
    """
    # 将 BGR 图像转换为 Luv 色彩空间
     # 降采样图像
    img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # 转换为 Luv 色彩空间
    img_luv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Luv)

    # 获取图像尺寸
    h, w, c = img_luv.shape
    # 将图像展开为特征矩阵，包含 Luv 和像素位置
    flat_img = img_luv.reshape((-1, c))
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    flat_features = np.hstack((
        flat_img,
        (x.reshape(-1, 1) / spatial_bandwidth),  # 缩放 x 坐标
        (y.reshape(-1, 1) / spatial_bandwidth)   # 缩放 y 坐标
    ))

    # 使用颜色特征（不包含位置特征）以减少维度
    flat_features = flat_img

    # 设置带宽
    bandwidth = range_bandwidth

    # Mean Shift 聚类
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    print("开始 Mean Shift 聚类...")
    mean_shift.fit(flat_features)
    print("Mean Shift 聚类完成！")


    # 获取聚类标签并重塑为图像形状
    labels = mean_shift.labels_
    seg = labels.reshape(h, w)

    # 获取分割区域数量
    segnum = len(np.unique(seg))
    print(f"分割完成，区域数量: {segnum}")
    # 过滤小区域
    # seg, segnum = enforce_minimum_region_area(seg, min_region_area)

    return seg, segnum

def edison_meanshift_segmentation(img, spatial_bandwidth=9, range_bandwidth=25, min_region_area=200):
    """
    使用 Mean Shift 方法進行圖像分割，與 `segment_image` 格式一致。
    
    參數:
        img: 輸入的 BGR 圖像 (numpy array, shape: [H, W, 3])
        spatial_bandwidth: 空間帶寬，用於位置權重調整 (int)
        range_bandwidth: 顏色帶寬，用於顏色權重調整 (int)
        min_region_area: 最小區域面積阈值 (int)
    
    返回:
        seg: 分割後的標籤矩陣 (numpy array, shape: [H, W])
        segnum: 分割區域數量
    """
    # 將圖像縮小以加速處理
    img_resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    img_luv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Luv)

    # 提取特徵：顏色 + 空間位置
    h, w, c = img_luv.shape
    flat_img = img_luv.reshape((-1, c))
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    flat_features = np.hstack((
        flat_img / range_bandwidth,
        x.reshape(-1, 1) / spatial_bandwidth,
        y.reshape(-1, 1) / spatial_bandwidth
    ))

    # 執行 Mean Shift 聚類
    print("開始 Mean Shift 聚類...")
    ms = MeanShift(bandwidth=1, bin_seeding=True)
    ms.fit(flat_features)
    print("Mean Shift 聚類完成！")

    # 重塑標籤矩陣
    labels = ms.labels_.reshape(h, w)

    # 濾除小區域
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    small_labels = unique_labels[label_counts < min_region_area]

    for small_label in small_labels:
        small_region_mask = (labels == small_label)
        neighbor_labels = np.unique(labels[small_region_mask])
        neighbor_labels = neighbor_labels[neighbor_labels != small_label]
        if len(neighbor_labels) > 0:
            mode_label = np.argmax(np.bincount(neighbor_labels))
            labels[small_region_mask] = mode_label

    segnum = len(np.unique(labels))

    return labels, segnum


def enforce_minimum_region_area(seg, min_area):
    """
    过滤掉分割结果中面积小于指定阈值的区域
    Args:
        seg: 分割图像矩阵 (numpy array, shape: [H, W])
        min_area: 最小区域面积阈值 (int)
    Returns:
        filtered_seg: 过滤后重新标记的分割图像
        segnum: 分割区域数量
    """
    # 标记连通区域
    labeled_seg, num_features = label(seg)

    # 计算每个区域的像素数量
    unique_labels, counts = np.unique(labeled_seg, return_counts=True)

    # 创建过滤后的分割图像
    filtered_seg = np.zeros_like(labeled_seg)

    # 只保留像素数量大于等于 min_area 的区域
    new_label = 1
    for label_val, count in zip(unique_labels, counts):
        if label_val == 0:  # 跳过背景
            continue
        if count >= min_area:
            filtered_seg[labeled_seg == label_val] = new_label
            new_label += 1

    segnum = new_label - 1
    return filtered_seg, segnum


def generate_filter_bank(scales, orientations):
    """
    Generate a Gabor filter bank with specified scales and orientations.
    Args:
        scales: List of frequencies (e.g., [0.1, 0.2, 0.3]).
        orientations: List of orientations in radians (e.g., [0, np.pi/4, np.pi/2]).
    Returns:
        filter_bank: List of (frequency, theta) tuples.
    """
    filter_bank = []
    for scale in scales:
        for theta in orientations:
            filter_bank.append((scale, theta))
    return filter_bank

def apply_gabor_filter_bank(gray_image, filter_bank):
    """
    應用 Gabor 濾波器組到灰度圖像
    Args:
        gray_image: 灰度圖像 (numpy array)
        filter_bank: Gabor 濾波器參數列表 (scale, theta)
    Returns:
        features: 濾波後的特徵 (h, w, n_filters)
    """
    h, w = gray_image.shape
    n_filters = len(filter_bank)
    features = np.zeros((h, w, n_filters))
    
    for idx, (scale, theta) in enumerate(filter_bank):
        filt_real, filt_imag = gabor(gray_image, frequency=scale, theta=theta)
        features[:, :, idx] = np.sqrt(filt_real**2 + filt_imag**2)  # 計算幅值
    return features

def assign_textons(features, n_clusters=128):
    """
    將濾波器輸出特徵分配到文字子集
    Args:
        features: 濾波特徵 (h, w, n_filters)
        n_clusters: 文字子集的數量
    Returns:
        assigned_indices: 每個像素的文字子集索引 (h, w)
    """
    h, w, n_filters = features.shape
    features_flat = features.reshape(-1, n_filters)  # 展平到 (n_pixels, n_filters)

    # 使用 K-means 聚類生成文字子集
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_flat)
    textons = kmeans.cluster_centers_  # 文字子集中心

    # 分配文字子集
    assigned_indices = kmeans.predict(features_flat)  # 分配索引
    return assigned_indices.reshape(h, w), textons

def remove_boundaries(img, seg):
    """
    將分割區域的邊界設置為黑色
    Args:
        img: 原始 RGB 圖像 (numpy array)
        seg: 分割標籤矩陣，每個像素對應區域標籤 (numpy array)
    Returns:
        nim: 處理後的圖像，邊界像素設為黑色
    """
    # 計算梯度，檢測分割邊界
    gx, gy = np.gradient(seg.astype(float))
    eim = (gx**2 + gy**2) > 1e-10  # 邊界掩碼

    # 將邊界設置為黑色
    nim = img.copy()
    nim[eim] = [0, 0, 0]  # 邊界像素設置為黑色

    return nim

def calc_texton_histogram(seg, assigned_indices, n_clusters):
    """
    計算每個區域的文字子集直方圖
    Args:
        seg: 分割標籤矩陣，每個像素的值是區域標籤
        assigned_indices: 每個像素的文字子集索引
        n_clusters: 文字子集的數量
    Returns:
        histograms: 每個區域的直方圖 (num_regions, n_clusters)
    """
    num_regions = np.max(seg) + 1  # 假設區域索引從 0 開始
    histograms = np.zeros((num_regions, n_clusters))
    
    for region in range(num_regions):  # 從 0 到 num_regions-1
        mask = (seg == region)  # 提取當前區域的掩碼
        region_indices = assigned_indices[mask]  # 提取該區域的文字子集索引
        hist, _ = np.histogram(region_indices, bins=np.arange(n_clusters + 1))
        histograms[region, :] = hist / (np.sum(hist) + 1e-6)  # 歸一化
    return histograms

# def visualize_nearest_connections(between, centroids):
    """
    找到每個區域最近的區域，並繪製重心之間的連接線
    Args:
        between: 區域之間的距離矩陣 (numpy array, shape: [segnum, segnum])
        centroids: 每個區域的重心 (numpy array, shape: [segnum, 2])
    """
    segnum = centroids.shape[0]
    near = np.zeros(segnum, dtype=int)  # 存儲每個區域最近的區域索引

    # 初始化圖像
    plt.figure()
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label="Centroids")  # 繪製重心點

    # 尋找最近的區域並繪製連接線
    for i in range(segnum):
        # 找到最近的區域
        value = np.min(between[i, :])  # 最近的距離值
        near[i] = np.argmin(between[i, :])  # 最近區域的索引
        j = near[i]

        # 繪製從區域 i 到最近區域 j 的連接線
        plt.plot(
            [centroids[i, 0], centroids[j, 0]], 
            [centroids[i, 1], centroids[j, 1]], 
            'b-', label="Connection" if i == 0 else ""
        )

        # 可選：在重心上標記座標
        # plt.text(centroids[i, 0], centroids[i, 1], f"({centroids[i, 0]:.1f}, {centroids[i, 1]:.1f})", fontsize=8)

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Nearest Connections between Regions")
    plt.show()
def visualize_nearest_connections(between, centroids, seg, image):
    """
    找到每個區域最近的區域，並在分割圖像背景上繪製重心之間的連接線
    Args:
        between: 區域之間的距離矩陣 (numpy array, shape: [segnum, segnum])
        centroids: 每個區域的重心 (numpy array, shape: [segnum, 2])
        seg: 分割後的圖像 (numpy array, shape: [H, W])
        image: 原始圖像 (numpy array, shape: [H, W, 3])
    """
    segnum = centroids.shape[0]
    near = np.zeros(segnum, dtype=int)  # 存儲每個區域最近的區域索引

    # 創建一個圖層作為背景
    plt.figure(figsize=(10, 10))
    plt.imshow(image)  # 原始圖像作為背景

    # 使用唯一值顯示分割區域（區域顏色加透明度）
    seg_overlay = np.zeros_like(image, dtype=float)
    for i in range(segnum):
        mask = seg == i   # seg 中的每個區域索引是 1-based
        color = np.random.rand(3)  # 隨機顏色
        seg_overlay[mask] = color  # 為該區域上色

    # 疊加分割區域
    plt.imshow(seg_overlay, alpha=0.5)

    # 繪製重心點
    plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=100, label="Centroids")  # 繪製重心點

    # 尋找最近的區域並繪製連接線
    for i in range(segnum):
        # 找到最近的區域
        value = np.min(between[i, :])  # 最近的距離值
        near[i] = np.argmin(between[i, :])  # 最近區域的索引
        j = near[i]

        # 繪製從區域 i 到最近區域 j 的連接線
        plt.plot(
            [centroids[i, 1], centroids[j, 1]], 
            [centroids[i, 0], centroids[j, 0]], 
            'b-', linewidth=2, label="Connection" if i == 0 else ""
        )

        # 在重心上標記座標
        plt.text(
            centroids[i, 1], centroids[i, 0],
            f"{i+1}", fontsize=8, color='yellow',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Nearest Connections between Regions with Segmentation Overlay")
    plt.axis('off')
    plt.show()


def calculate_color_ratios(hsv, ycbcr, hsi, near):
    """
    計算每個區域與最近鄰區域之間的顏色相關比值特徵
    Args:
        hsv: 每個區域的 HSV 特徵 (numpy array, shape: [segnum, 3])
        ycbcr: 每個區域的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        hsi: 每個區域的 HSI 特徵 (numpy array, shape: [segnum, 3])
        near: 每個區域最近鄰的索引 (numpy array, shape: [segnum])
    Returns:
        hh: 比值特徵矩陣 (numpy array, shape: [3, segnum])
    """
    segnum = len(near)
    hh = np.zeros((3, segnum))  # 初始化特徵矩陣

    for i in range(segnum):
        j = near[i]  # 最近鄰的索引

        # 計算 HSV 第三分量（亮度值）的最大最小比值
        max_hsv = max(hsv[i, 2], hsv[j, 2])
        min_hsv = min(hsv[i, 2], hsv[j, 2])
        # hh[0, i] = min_hsv / max_hsv  # 如果需要啟用 HSV 比值

        # 計算 YCbCr 第一分量（亮度值）的最大最小比值
        max_ycbcr = max(ycbcr[i, 0], ycbcr[j, 0])
        min_ycbcr = min(ycbcr[i, 0], ycbcr[j, 0])
        hh[1, i] = min_ycbcr / max_ycbcr

        # 計算 HSI 特徵的 H/I 比值
        hsi_i_ratio = (hsi[i, 0] + 1/255) / (hsi[i, 2] + 1/255)
        hsi_j_ratio = (hsi[j, 0] + 1/255) / (hsi[j, 2] + 1/255)
        max_hsi = max(hsi_i_ratio, hsi_j_ratio)
        min_hsi = min(hsi_i_ratio, hsi_j_ratio)
        hh[0, i] = min_hsi / max_hsi

        # 註釋掉的第三種比值（例如基於其他特徵的比較）
        # hh[2, i] = ll[i] == ll[j]

    return hh

def cluster_hsi_ratios(hsi):
    """
    對 HSI 特徵的 H/I 比值進行聚類，計算聚類中心與標準差
    Args:
        hsi: HSI 特徵矩陣 (numpy array, shape: [segnum, 3])
    Returns:
        center: 聚類中心 (numpy array, shape: [2])
        c_std: 每個聚類的標準差 (numpy array, shape: [2])
    """
    # 計算 H/I 比值
    x = (hsi[:, 0] + 1e-8) / (hsi[:, 2] + 1e-8)  # 防止除零
    x = x.reshape(-1, 1)  # 將數據展平為列向量

    # K-means 聚類
    kmeans = KMeans(n_clusters=2, random_state=42)
    idx = kmeans.fit_predict(x)  # 聚類標籤
    center = kmeans.cluster_centers_.flatten()  # 聚類中心

    # 計算每個聚類的標準差
    c_std = np.zeros(2)
    for cluster in range(2):
        cluster_points = x[idx == cluster]
        c_std[cluster] = np.std(cluster_points)

    # 對聚類中心和標準差進行排序
    if center[0] > center[1]:
        center = np.sort(center)
        c_std = c_std[::-1]  # 同時調整標準差順序

    return center, c_std

def detect_shadow_regions(ycbcr, hsi, segnum):
    """
    檢測陰影區域並標記
    Args:
        ycbcr: 每個區域的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        hsi: 每個區域的 HSI 特徵 (numpy array, shape: [segnum, 3])
        segnum: 區域數量 (int)
    Returns:
        label: 陰影區域標籤 (numpy array, shape: [segnum])
        ycbcr_copy: 修改後的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        n_nonshadow: 非陰影區域數量 (int)
        flag: 檢測到的陰影區域數量 (int)
    """
    # 初始化
    label = np.ones(segnum) * 255  # 初始化標籤，255 表示未分類
    ycbcr_copy = ycbcr.copy()  # 副本，用於清空陰影區域
    n_nonshadow = segnum  # 非陰影區域數量
    avg_y = np.mean(ycbcr[:, 0])  # Y 分量平均值
    flag = 0  # 陰影區域數量

    # 計算 HSI 比值
    t_hsi = (hsi[:, 0] + 1e-8) / (hsi[:, 2] + 1e-8)  # 防止除零
    level = threshold_otsu(t_hsi)  # Otsu 阈值

    # 判定陰影區域
    for i in range(segnum):
        if ycbcr[i, 0] < avg_y * 0.6:  # 判斷是否為陰影區域
            label[i] = 0  # 標記為陰影
            ycbcr_copy[i, :] = 0  # 清空 YCbCr 值
            n_nonshadow -= 1  # 減少非陰影區域數量
            flag += 1  # 增加陰影區域數量

    return label, ycbcr_copy, n_nonshadow, flag

def iterative_shadow_detection(hsi, ycbcr, label, near, center, c_std, segnum, n_nonshadow):
    """
    使用迭代方法檢測陰影區域
    Args:
        hsi: 每個區域的 HSI 特徵 (numpy array, shape: [segnum, 3])
        ycbcr: 每個區域的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        label: 區域標籤 (numpy array, shape: [segnum])
        near: 每個區域的最近鄰 (numpy array, shape: [segnum])
        center: 聚類中心 (numpy array, shape: [2, 1])
        c_std: 聚類標準差 (numpy array, shape: [2, 1])
        segnum: 區域數量 (int)
        n_nonshadow: 非陰影區域數量 (int)
    Returns:
        label: 更新後的標籤 (numpy array, shape: [segnum])
        ycbcr: 更新後的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        n_nonshadow: 更新後的非陰影區域數量 (int)
        refuse: 更新後的拒絕標記 (numpy array, shape: [segnum])
    """
    refuse = np.zeros(segnum)  # 初始化拒絕標記
    flag = 0  # 記錄陰影區域數量

    while True:
        update = 0
        new = 0
        max_v = 0

        # 遍歷所有區域，尋找最符合陰影條件的區域
        for i in range(segnum):
            val = hsi[i, 0] / (hsi[i, 2] + 1e-8)  # 計算 H/I 比值
            temp1 = norm.cdf((val - center[1]) / c_std[1])  # 第二聚類正態分佈概率
            temp2 = norm.cdf(-(val - center[0]) / c_std[0])  # 第一聚類正態分佈概率

            # 判斷是否滿足陰影條件
            if temp2 < temp1 and refuse[i] == 0 and label[i] == 255:
                if temp1 > max_v:
                    new = i  # 更新為最符合條件的區域
                    max_v = temp1
                    update = 1

        # 如果沒有更新或匹配概率低於閾值，結束迭代
        if update == 0 or max_v < 0.0028:
            break

        # 標記新區域為陰影
        label[new] = 0
        j = near[new]  # 找到最近鄰區域
        vali = hsi[new, 0] / (hsi[new, 2] + 1e-8)
        valj = hsi[j, 0] / (hsi[j, 2] + 1e-8)

        # 判斷是否拒絕將最近鄰區域標記為陰影
        if ((vali - center[1]) / c_std[1]) - ((valj - center[1]) / c_std[1]) > 3:
            refuse[j] = 1  # 拒絕標記
            label[j] = 255  # 還原最近鄰標籤為未分類狀態

        # 更新 YCbCr 特徵與區域計數
        ycbcr[new, :] = 0
        n_nonshadow -= 1
        flag += 1

    return label, ycbcr, n_nonshadow, refuse

def update_shadow_labels(label, near, hsv, ycbcr, hh, seg):
    """
    基於相似性更新陰影標籤
    Args:
        label: 區域標籤 (numpy array, shape: [segnum])
        near: 每個區域的最近鄰 (numpy array, shape: [segnum])
        hsv: 每個區域的 HSV 特徵 (numpy array, shape: [segnum, 3])
        ycbcr: 每個區域的 YCbCr 特徵 (numpy array, shape: [segnum, 3])
        hh: 額外特徵 (numpy array, shape: [1, segnum])
        seg: 分割圖像矩陣 (numpy array, shape: [height, width])
    Returns:
        label: 更新後的標籤
    """
    segnum = len(label)
    
    for i in range(segnum):
        if label[i] != 255:  # 略過已標記的區域
            continue
        
        j = near[i]  # 最近鄰索引

        # 計算 HSV 和 YCbCr 特徵的最大值與最小值
        max_hsv = max(hsv[i, 2], hsv[j, 2])
        min_hsv = min(hsv[i, 2], hsv[j, 2])
        max_ycbcr = max(ycbcr[i, 0], ycbcr[j, 0])
        min_ycbcr = min(ycbcr[i, 0], ycbcr[j, 0])

        # 計算相似性指標
        same = (min_hsv / (max_hsv + 1e-8) + 
                min_ycbcr / (max_ycbcr + 1e-8) + 
                hh[0, i])

        # 判定是否標記為陰影
        if same > 2.5 and label[j] == 0:
            print(f"hh[1, {i}]: {hh[0, i]}, same: {same}")
            label[i] = 0

    return label

# extract features from each segment
def calculate_features(img,seg,segnum):
    """
    計算每個區域的特徵，包括 HSI、HSV、YCbCr、梯度、紋理和形狀。
    Args:
        image: 輸入 RGB 圖像 (numpy array)
        seg: 分割後的標籤矩陣，每個像素對應區域標籤
        segnum: 分割區域數量
    Returns:
        features: 特徵字典，包括 HSI, HSV, YCbCr, 梯度, 紋理等
    """
    h, w, _ = img.shape
    seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    epsilon = 1e-8

    # 初始化特徵
    hsi = np.zeros((segnum, 3))  # HSI 特徵
    hsv = np.zeros((segnum, 3))  # HSV 特徵
    ycbcr = np.zeros((segnum, 3))  # YCbCr 特徵
    grad = np.zeros((segnum, 2))  # 梯度特徵 (水平與垂直梯度)
    texthist = np.zeros((segnum, 128))  # 紋理特徵（基於直方圖）
    centroids = np.zeros((segnum, 2))  # 每個區域的中心點
    area = np.zeros(segnum)  # 每個區域的面積
    refuse = np.zeros(segnum)  # 拒絕標記

    # 計算每個區域的特徵
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsi_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 生成 Gabor 濾波器組
    scales = [0.1, 0.2, 0.3]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    filter_bank = generate_filter_bank(scales, orientations)
    filtered_features = apply_gabor_filter_bank(gray_img, filter_bank)
    assigned_indices, _ = assign_textons(filtered_features)  # 返回整幅圖像的索引矩陣 (h, w)

    # 使用 regionprops 計算形狀特徵
    properties = regionprops(seg)

    # traverse each segment get the features
    for i in range(segnum):
        # 提取第 i 個區域的像素索引
        mask = (seg == i)

        # 計算 HSI、HSV、YCbCr 特徵
        hsi[i, :] = np.mean(hsi_img[mask], axis=0)
        hsv[i, :] = np.mean(hsv_img[mask], axis=0)
        ycbcr[i, :] = np.mean(ycbcr_img[mask], axis=0)

        # 計算texture features
        # 提取當前區域的文字子集索引
        region_indices = assigned_indices[mask]
        hist, _ = np.histogram(region_indices, bins=np.arange(129))  # bins範圍 0 到 128
        texthist[i, :] = hist / (np.sum(hist) + epsilon)  # 歸一化
        # print("文字子集直方圖的形狀：", texthist.shape)

        # 計算梯度特徵
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
        gmag = np.sqrt(grad_x**2 + grad_y**2)  # 梯度幅值
        gdir = np.arctan2(grad_y, grad_x)  # 梯度方向（可選）
        region_gmag = gmag[mask]
        grad[i, 0] = np.mean(region_gmag)  # 該區域內梯度幅值的均值
        grad[i, 1] = np.std(region_gmag)  # 該區域內梯度幅值的標準差

        # 計算形狀特徵
        for region in properties:
            if region.label == i + 1:  # 注意 regionprops 的 label 是 1-based 索引
                area[i] = region.area  # 區域面積
                centroids[i, :] = region.centroid  # 區域重心

    # normalize features
    ycbcr[:, 0] /= np.max(ycbcr[:, 0] + epsilon) # 歸一化 Y
    hsv[:, 0] /= np.max(hsv[:, 0] + epsilon) # 歸一化 H

    # calculate Dij(si and sj)
    between = np.zeros((segnum, segnum))
    for i in range(segnum):
        for j in range(segnum):
            if i == j:
                between[i, j] = 100 # 本身跟本身的距離設為100 避免Dij=0
                continue
            # 計算重心之間的歐幾里得距離
            distance = np.sqrt(np.sum((centroids[i, :] - centroids[j, :])**2))
            # 計算梯度差異的 L1 距離
            grad_distance = np.sum(np.abs(grad[i, :] - grad[j, :]))
            # 計算紋理直方圖差異的 L1 距離
            texthist_distance = np.sum(np.abs(texthist[i, :] - texthist[j, :]))
            # Dij = Dgrad + Dtexthist + Ddistance
            between[i, j] = grad_distance + texthist_distance + distance
    
    # remove the boundary
    nim = remove_boundaries(img, seg)
    plt.imshow(nim)
    plt.title("Processed Image with Edges Highlighted")
    plt.show()

    # visualize the nearest connections
    visualize_nearest_connections(between, centroids,seg,nim)

    # save the ration of the area of the segment feature
    ratio_hh = calculate_color_ratios(hsv, ycbcr, hsi, np.argmin(between, axis=1))

    # cluster the H/I ratios
    center, c_std = cluster_hsi_ratios(hsi)

    #label the shadow regions
    shadow_labels, ycbcr_copy, n_nonshadow, flag = detect_shadow_regions(ycbcr, hsi, segnum)

    # refuse the shadow regions
    shadow_labels, ycbcr_copy, n_nonshadow, refuse = iterative_shadow_detection(hsi, ycbcr, shadow_labels, np.argmin(between, axis=1), center, c_std, segnum, n_nonshadow)

    # update the shadow labels
    shadow_labels = update_shadow_labels(shadow_labels, np.argmin(between, axis=1), hsv, ycbcr, ratio_hh, seg)

    plt.imshow(seg, cmap='gray')
    plt.title("Updated Labels")
    plt.show()

    # return the features
    features = {
        "hsi": hsi,
        "hsv": hsv,
        "ycbcr": ycbcr,
        "grad": grad,
        "texthist": texthist,
        "centroids": centroids,
        "area": area,
        "refuse": refuse,
        "shadow_labels": shadow_labels
    }
    return features


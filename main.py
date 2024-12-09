import cv2
import numpy as np
import os
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from skimage.segmentation import mark_boundaries, quickshift
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from scipy.stats import norm


def quick_shift(img): 
    # 將 BGR 圖像轉換為 LAB 色彩空間
    LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 使用 Quickshift 分割圖像
    seg = quickshift(
        LAB_img,
        kernel_size=15,    # 對應 MATLAB 的 SpatialBandWidth
        max_dist=15,      # 對應 MATLAB 的 RangeBandWidth
        ratio=0.1       # 可調參數，用於控制區域間距離
    )
    # 獲取區域數量
    segnum = np.max(seg)  # 最大標籤值即為區域數量

    border_img = (mark_boundaries(img, seg, color=(0, 0, 0)) * 255).astype(np.uint8)

    return seg, border_img


def mean_shift(img):

    height, width, _ = img.shape
    img = cv2.medianBlur(img, 3)
    flat_img = img.reshape((-1, 3))

    # Create the feature space
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    flat_img_with_coordinates = np.column_stack([flat_img, x.flatten(), y.flatten()])

    # Estimate bandwidth for Mean Shift
    bandwidth = estimate_bandwidth(flat_img_with_coordinates, quantile=0.01, n_samples=500)

    # Perform Mean Shift clustering
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    print("mean_shift.fit...")
    mean_shift.fit(flat_img_with_coordinates)
    labels = mean_shift.labels_

    # Reshape the labels to the original image shape
    segmented_img = labels.reshape((height, width))

    # Generate a segmented image
    """
    unique_labels = np.unique(labels)
    segmented_colors = np.random.randint(1, 255, size=(len(unique_labels), 3))
    colored_segmented_img = segmented_colors[segmented_img]
    colored_segmented_img = colored_segmented_img.astype(np.uint8)
    """
    border_img = (mark_boundaries(img, segmented_img, color=(0, 0, 0)) * 255).astype(np.uint8)

    return segmented_img, border_img

def mean_shift_with_merge(img, min_region_size=800):
    """
    Perform Mean Shift segmentation and merge small regions.
    
    :param img: Input image
    :param min_region_size: Minimum size of a region to retain. Smaller regions will be merged.
    :return: segmented_img (label image), border_img (image with boundaries marked)
    """

    height, width, _ = img.shape
    img = cv2.medianBlur(img, 3)
    flat_img = img.reshape((-1, 3))

    # Create the feature space
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    flat_img_with_coordinates = np.column_stack([flat_img, x.flatten(), y.flatten()])

    # Estimate bandwidth for Mean Shift
    bandwidth = estimate_bandwidth(flat_img_with_coordinates, quantile=0.01, n_samples=500)

    # Perform Mean Shift clustering
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    print("mean_shift.fit...")
    mean_shift.fit(flat_img_with_coordinates)
    labels = mean_shift.labels_

    # Reshape the labels to the original image shape
    segmented_img = labels.reshape((height, width))

    # Merge small regions
    unique_labels, counts = np.unique(segmented_img, return_counts=True)
    small_regions = unique_labels[counts < min_region_size]

    for region_label in small_regions:
        mask = segmented_img == region_label

        # Find the nearest neighboring region
        dilated_mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        neighbor_labels = segmented_img[dilated_mask.astype(bool) & ~mask]
        neighbor_labels = neighbor_labels[neighbor_labels != region_label]  # Exclude itself

        if len(neighbor_labels) > 0:
            # Replace the region_label with the most common neighboring label
            most_common_label = np.bincount(neighbor_labels).argmax()
            segmented_img[mask] = most_common_label

    # Relabel to ensure labels are continuous
    unique_labels = np.unique(segmented_img)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    relabeled_img = np.vectorize(label_mapping.get)(segmented_img)

    # Generate the boundary image
    border_img = (mark_boundaries(img, relabeled_img, color=(0, 0, 0)) * 255).astype(np.uint8)

    return relabeled_img, border_img


def find_center(img, segmented_img):

    center_marked_img = img.copy()

    center_indices = []
    label_num = len(np.unique(segmented_img))
    for i in range(label_num):
        center = np.mean(np.argwhere(segmented_img == i), axis=0)
        center_y, center_x = int(round(center[0])), int(round(center[1]))
        center_marked_img[center_y, center_x] = [255, 255, 0]
        center_indices.append([center_y, center_x])
    center_indices = np.array(center_indices)

    center_marked_img = (mark_boundaries(center_marked_img, segmented_img, color=(0, 0, 0)) * 255).astype(np.uint8)

    return center_indices, center_marked_img

def histogram(traget_img, segmented_img, binNum):
 
    label_num = len(np.unique(segmented_img))

    # Set up bins for the gradient magnitudes
    inter = np.max(traget_img) / binNum
    traget_img = (traget_img / inter).astype(np.int32)
    
    binVal = np.arange(0, binNum)
    desc = np.zeros((label_num, binNum))
    
    # Prepare indices for regions
    ind = {}
    for iReg in range(0, label_num):
        ind[iReg] = (segmented_img.ravel() == iReg)
    
    # Calculate the descriptor
    for cnt, bin_val in enumerate(binVal):
        I = (traget_img.ravel() == bin_val)
        for iReg in range(0, label_num):
            desc[iReg, cnt] = np.sum(I[ind[iReg]])
    
    # Normalize the descriptor
    tmp = np.sum(desc, axis=1, keepdims=True)
    desc = desc / (tmp + 1e-10)  # Add small epsilon to prevent division by zero
    
    return desc


def cal_gradient_hist(img, segmented_img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) 

    return histogram(grad_mag, segmented_img, 20), grad_mag
   

def cal_texture_hist(img, segmented_img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # LBP 參數
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp_norm = np.interp(lbp, (lbp.min(), lbp.max()), (0, 255)).astype(np.uint8)
    
    return histogram(lbp_norm, segmented_img, 128), lbp_norm

def cal_color_hist(img, segmented_img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab_img[:, :, 0]
    A = lab_img[:, :, 1]
    B = lab_img[:, :, 2]
    
    # Normalize the color channels
    L = np.interp(L, (L.min(), L.max()), (0, 255)).astype(np.uint8)
    A = np.interp(A, (A.min(), A.max()), (0, 255)).astype(np.uint8)
    B = np.interp(B, (B.min(), B.max()), (0, 255)).astype(np.uint8)


    return histogram(A+B, segmented_img, 36)



def apply_gabor_filters(gray_img, scales=5, orientations=8):
    """
    Apply Gabor filters with multiple scales and orientations to the grayscale image.
    """
    gabor_results = []
    for scale in range(scales):
        for theta in range(orientations):
            theta_angle = theta * np.pi / orientations
            sigma = 4.0  # Standard deviation of the Gaussian function
            lambd = 10.0  # Wavelength of the sinusoidal factor
            gamma = 0.5  # Spatial aspect ratio
            kernel_size = 21  # Kernel size
            psi = 0  # Phase offset
            
            # Create Gabor kernel
            gabor_kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size), sigma, theta_angle, lambd, gamma, psi, ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered_img = cv2.filter2D(gray_img, cv2.CV_32F, gabor_kernel)
            gabor_results.append(filtered_img)
    
    return gabor_results

def cal_texture_hist_gabor(img, segmented_img):
    """
    Calculate texture histogram using Gabor filters.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gabor filters
    gabor_filtered_imgs = apply_gabor_filters(gray_img, 5, 8)
    
    # Combine all Gabor responses by summing their magnitudes
    combined_response = np.sum(np.abs(np.array(gabor_filtered_imgs)), axis=0)
    
    # Normalize combined response
    response_norm = np.interp(combined_response, (combined_response.min(), combined_response.max()), (0, 255)).astype(np.uint8)
    
    return histogram(response_norm, segmented_img, 128), response_norm

def cal_center_dist(img, i, j, center_indices):
    height, width, _ = img.shape
    dist = np.sum((center_indices[i] - center_indices[j]) ** 2) ** 0.5
    dist_norm = dist / max(height, width)
    #print("dist_norm[{}, {}]".format(i, j), dist_norm)
    return dist_norm

def find_nearest_region(img, segmented_img, center_indices):
    label_num = len(np.unique(segmented_img)) 
    near_labels = np.full((label_num), -1)
    # print("near_labels.shape", near_labels.shape)

    gradient_hist, grad_mag = cal_gradient_hist(img, segmented_img)
    texture_hist, lbp_norm = cal_texture_hist(img, segmented_img)
    
    for i in range(label_num):
        distances = []
        for j in range(label_num):
            gradient_dist = np.sum(np.abs(gradient_hist[i] - gradient_hist[j]))
            texture_dist = np.sum(np.abs(texture_hist[i] - texture_hist[j]))
            center_dist = cal_center_dist(img, i, j, center_indices)
            distance = gradient_dist + texture_dist + center_dist
            distances.append(distance)
        distances[i] = max(distances) + 1
        near_label = distances.index(min(distances))
        near_labels[i] = near_label
    # print("near_labels", near_labels)

    return near_labels, grad_mag, lbp_norm

def draw_nearest_region(img, segmented_img, center_indices, near_labels):
    label_num = len(np.unique(segmented_img))
    near_img = (mark_boundaries(img, segmented_img, color=(0, 0, 0)) * 255).astype(np.uint8)
    for i in range(label_num):
        center_index_i =  tuple(np.flip(center_indices[i]))
        center_index_near_i = tuple(np.flip(center_indices[near_labels[i]]))
        cv2.line(near_img, center_index_i, center_index_near_i, (255, 0, 0))
    return near_img

def draw_nearest_region_only_shadow(img, segmented_img, center_indices, near_labels,labels):
    label_num = len(np.unique(segmented_img))
    near_img = (mark_boundaries(img, segmented_img, color=(0, 0, 0)) * 255).astype(np.uint8)
    for i in range(label_num):
        if labels[i] == 0:
            center_index_i =  tuple(np.flip(center_indices[i]))
            center_index_near_i = tuple(np.flip(center_indices[near_labels[i]]))
            cv2.line(near_img, center_index_i, center_index_near_i, (255, 0, 0))
    return near_img

def cal_region_avg_R(img, segmented_img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, v_img = hsv_img[:, :, 0], hsv_img[:, :, 2]
    h_norm = np.interp(h_img, (h_img.min(), h_img.max()), (0, 1))
    v_norm = np.interp(v_img, (v_img.min(), v_img.max()), (0, 1))
    r = h_norm / (v_norm + 1e-10)  # Add small epsilon to prevent division by zero

    label_num = len(np.unique(segmented_img))

    # Remove extreme values
    segmented_img_remove_extreme = segmented_img.copy()
    segmented_img_remove_extreme[v_norm == 0] = -1
    
    # Prepare indices for regions
    ind = {}
    for iReg in range(0, label_num):
        ind[iReg] = (segmented_img_remove_extreme.ravel() == iReg)
    
    # Calculate average R value for each region
    r = r.ravel()
    R = np.zeros((label_num))
    for iReg in range(0, label_num):
        R[iReg] = np.sum(r[ind[iReg]]) / np.sum(ind[iReg])

    return R

def cal_region_avg_Y(img, segmented_img):
    
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    label_num = len(np.unique(segmented_img))

    # Prepare indices for regions
    ind = {}
    for iReg in range(0, label_num):
        ind[iReg] = (segmented_img.ravel() == iReg)
    
    # Calculate average Y value for each region
    y = ycrcb_img[:, :, 0].ravel()
    Y = np.zeros((label_num))
    for iReg in range(0, label_num):
        Y[iReg] = np.sum(y[ind[iReg]]) / np.sum(ind[iReg])

    return Y


def cal_region_avg_H(img, segmented_img):
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    label_num = len(np.unique(segmented_img))

    # Prepare indices for regions
    ind = {}
    for iReg in range(0, label_num):
        ind[iReg] = (segmented_img.ravel() == iReg)
    
    # Calculate average H value for each region
    h = hsv_img[:, :, 0].ravel()
    H = np.zeros((label_num))
    for iReg in range(0, label_num):
        H[iReg] = np.sum(h[ind[iReg]]) / np.sum(ind[iReg])

    return H

def shadow_light_cluster(img, segmented_img):

    R = cal_region_avg_R(img, segmented_img)

    model = KMeans(n_clusters=2, random_state=1)
    model.fit(R.reshape(-1, 1))
    cluster_labels = model.labels_
    cluster_centers = model.cluster_centers_

    cluster_std = np.zeros_like(cluster_centers)
    cluster_std[0] = np.std(R.ravel()[cluster_labels == 0])
    cluster_std[1] = np.std(R.ravel()[cluster_labels == 1])

    # Ensure cluster_centers[0] is Clit and cluster_centers[1] is Cshadow (Clit =< Cshadow)
    if(cluster_centers[0] > cluster_centers[1]):
        cluster_centers[[0, 1]] = cluster_centers[[1, 0]]
        cluster_std[[0, 1]] = cluster_std[[1, 0]]
        cluster_labels = cluster_labels ^ 1

    # print("np.hstack(R.ravel(), c)")
    # print(np.transpose(np.vstack([R.ravel(), cluster_labels])))
    # print("cluster_centers =", cluster_centers)
    # print("cluster_std =", cluster_std)

    #plt.scatter(R.ravel(), np.zeros_like(R.ravel()), c=cluster_labels, s=10)
    #plt.scatter(cluster_centers, np.zeros_like(cluster_centers), c='red', s=30)
    #plt.show()

    # Mark the shadow region calculated by KMeans in the image
    shadow_indices = np.where(cluster_labels == 1)[0]
    shadow_mask = np.zeros_like(segmented_img)
    for i in shadow_indices:
        shadow_mask[segmented_img == i] = 1
    kmeans_img = img.copy()
    kmeans_img[np.where(shadow_mask == 1)] = np.array([0, 0, 255])

    return cluster_centers, cluster_std, kmeans_img


def shadow_detection(img, segmented_img, cluster_centers, cluster_std, near_labels):
    
    label_num = len(np.unique(segmented_img))
    
    Y_region = cal_region_avg_Y(img, segmented_img)
    R_region = cal_region_avg_R(img, segmented_img)
    H_region = cal_region_avg_H(img, segmented_img)
    label = np.ones((label_num))

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_mean = np.mean(ycrcb_img[:, :, 0])
    # print("Y_mean", Y_mean)

    # 2) 若 Yi < 60% ∗ mean(Yimage)，则 labeli = shadow
    for iReg in range(label_num):
        if(Y_region[iReg] < Y_mean * 0.6):
            label[iReg] = 0     # 0 = shadow, 1 = not shadow

    refuse = np.zeros((label_num))

    # 5) 反复迭代执⾏步骤 3)-4)，直到不再有更新发⽣；
    while(1):
        # 3) 选取 Fshadow 最⼤且 Refusei = 0 的区域 Si，设置 labeli = shadow；
        update = False
        max_F_shadow = 0
        max_F_shadow_label = 0
        for iReg in range(label_num):
            F_shadow = norm.cdf((R_region[iReg] - cluster_centers[1]) / cluster_std[1])
            F_lit = norm.cdf(-(R_region[iReg] - cluster_centers[0]) / cluster_std[0])
            if(F_lit < F_shadow and refuse[iReg] == 0 and label[iReg] == 1):
                if(F_shadow > max_F_shadow):
                    update = True
                    max_F_shadow = F_shadow
                    max_F_shadow_label = iReg
        # print("max_F_shadow =", max_F_shadow)            
        # print("max_F_shadow_label =", max_F_shadow_label)
        if(not update): #if update == False || max_F_shadow < 0.0028
            # print("no update: break")
            break
        # print("update")
        label[max_F_shadow_label] = 0

        # 4) 记 Si 最近区域 Neari 为 Sj，通过⽐较 Ri, Rj，检查 Si 与 Sj 是否为光亮相反区域，如果是，判断 Refusej = 1；
        i = max_F_shadow_label
        j = near_labels[max_F_shadow_label]
        z_i = (R_region[i] - cluster_centers[1]) / cluster_std[1]
        z_j = (R_region[j] - cluster_centers[1]) / cluster_std[1]
        if(z_i - z_j > 3):
            refuse[j] = 1
            label[j] = 1

    # 6) 对于 labeli = shadow 的 Si，通过⽐较 Yi, Yj , Ri, Rj，如果判断 Si 与 Sj 光亮类似，且Refusej = 0，设置 labelj = 0。
    for i in range(label_num):
        if(label[i] == 0):
            j = near_labels[i]
            min_H = min(H_region[i], H_region[j])
            max_H = max(H_region[i], H_region[j])
            min_Y = min(Y_region[i], Y_region[j])
            max_Y = max(Y_region[i], Y_region[j])
            min_R = min(R_region[i], R_region[j])
            max_R = max(R_region[i], R_region[j])
            if(min_H / max_H + min_Y / max_Y + min_R / max_R > 2.5 and refuse[j] == 0):
                label[j] == 0

    #visualization
    shadow_indices = np.where(label == 0)[0]
    shadow_mask = np.zeros_like(segmented_img)
    for i in shadow_indices:
        shadow_mask[segmented_img == i] = 1
    shadow_detect_img = img.copy()
    shadow_detect_img[np.where(shadow_mask == 1)] = np.array([255, 0, 0])

    return label, shadow_detect_img

def hist_match(source, template):
    """
    Match the histogram of the source region to that of the template region.
    
    Parameters:
    - source: ndarray
        The pixel values in the shadow region.
    - template: ndarray
        The pixel values in the non-shadow region.

    Returns:
    - matched: ndarray
        The pixel values of the source adjusted to match the template.
    """
    # 檢查是否有足夠像素進行匹配
    if len(source) == 0 or len(template) == 0:
        print("Empty source or template region. Skipping histogram matching.")
        return source  # 原樣返回

    # 增加微小隨機噪聲以避免單一值問題
    if np.ptp(source) == 0:  # 如果 source 是單一值（ptp: 最大值 - 最小值）
        source = source + np.random.normal(scale=0.01, size=source.shape)
    if np.ptp(template) == 0:  # 如果 template 是單一值
        template = template + np.random.normal(scale=0.01, size=template.shape)

    # 計算 source 和 template 的直方圖和累積分佈函數（CDF）
    src_values, bin_idx, src_counts = np.unique(source, return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    tmpl_cdf = np.cumsum(tmpl_counts).astype(np.float64) / template.size

    # 對應 source 的 CDF 到 template 的值
    interp_tmpl_values = np.interp(src_cdf, tmpl_cdf, tmpl_values)

    # 將對應值回映射到原圖
    matched = interp_tmpl_values[bin_idx]

    # 確保結果在有效範圍內（如 0~255）
    matched = np.clip(matched, np.min(template), np.max(template))

    return matched


import numpy as np

def find_non_shadow_pair(img, label, segmented_img, center_indices):
    label_num = len(np.unique(segmented_img))  # 區域數量
    near_labels = np.full((label_num), -1)    # 配對結果初始化為 -1

    # 計算梯度與紋理直方圖
    gradient_hist, grad_mag = cal_gradient_hist(img, segmented_img)
    texture_hist, lbp_norm = cal_texture_hist(img, segmented_img)

    # 圖像尺寸
    img_height, img_width = img.shape[:2]
    max_dim = max(img_height, img_width)  # 最大邊長，用於距離歸一化

    for i in range(label_num):
        if label[i] == 1:  # 跳過非陰影區域
            continue
        distances = []
        valid_indices = []

        for j in range(label_num):
            if label[j] == 1:  # 僅考慮非陰影區域
                # 計算梯度距離（絕對差值）
                gradient_dist = np.sum(np.abs(gradient_hist[i] - gradient_hist[j]))

                # 計算紋理距離（絕對差值）
                texture_dist = np.sum(np.abs(texture_hist[i] - texture_hist[j]))

                # 計算幾何中心距離，並進行歸一化
                center_i = center_indices[i]
                center_j = center_indices[j]
                euclidean_dist = np.sqrt((center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2)
                center_dist = euclidean_dist / max_dim

                # 總距離
                total_distance = gradient_dist + texture_dist + center_dist
                distances.append(total_distance)
                valid_indices.append(j)

        # 如果找到有效的非陰影區域，選擇距離最小的作為匹配對象
        if valid_indices:
            nearest_index = np.argmin(distances)
            near_labels[i] = valid_indices[nearest_index]

    return near_labels
def find_non_shadow_pair_v2(img, label, segmented_img, center_indices):
    label_num = len(np.unique(segmented_img))  # Number of regions
    near_labels = np.full((label_num), -1)    # Initialize pair results to -1

    # Calculate gradient, texture, and color histograms
    gradient_hist, grad_mag = cal_gradient_hist(img, segmented_img)
    texture_hist, lbp_norm = cal_texture_hist(img, segmented_img)
    color_hist = cal_color_hist(img, segmented_img)  # New color feature

    # Image dimensions
    img_height, img_width = img.shape[:2]
    max_dim = max(img_height, img_width)  # Max side length for distance normalization

    # Initialize min and max values for each feature
    gradient_min, gradient_max = float('inf'), float('-inf')
    texture_min, texture_max = float('inf'), float('-inf')
    center_min, center_max = float('inf'), float('-inf')
    color_min, color_max = float('inf'), float('-inf')  # For color feature

    # First pass: Compute all distances and find min/max values for normalization
    all_distances = {}  # To store intermediate distances for re-use
    for i in range(label_num):
        if label[i] == 1:  # Skip non-shadow regions
            continue
        for j in range(label_num):
            if label[j] == 1:  # Only consider non-shadow regions
                # Compute distances for each feature
                gradient_dist = np.sum(np.abs(gradient_hist[i] - gradient_hist[j]))
                texture_dist = np.sum(np.abs(texture_hist[i] - texture_hist[j]))
                center_i = center_indices[i]
                center_j = center_indices[j]
                euclidean_dist = np.sqrt((center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2)
                center_dist = euclidean_dist / max_dim
                color_dist = np.sum(np.abs(color_hist[i] - color_hist[j]))  # New color distance

                # Update min and max values for normalization
                gradient_min = min(gradient_min, gradient_dist)
                gradient_max = max(gradient_max, gradient_dist)
                texture_min = min(texture_min, texture_dist)
                texture_max = max(texture_max, texture_dist)
                center_min = min(center_min, center_dist)
                center_max = max(center_max, center_dist)
                color_min = min(color_min, color_dist)
                color_max = max(color_max, color_dist)

                # Store distances for re-use
                all_distances[(i, j)] = (gradient_dist, texture_dist, center_dist, color_dist)

    # Helper function to normalize a feature
    def normalize(value, min_val, max_val):
        if max_val - min_val == 0:  # Avoid division by zero
            return 0
        return (value - min_val) / (max_val - min_val)

    # Second pass: Compute total normalized distance and find the nearest region
    for i in range(label_num):
        if label[i] == 1:  # Skip non-shadow regions
            continue
        distances = []
        valid_indices = []

        for j in range(label_num):
            if label[j] == 1:  # Only consider non-shadow regions
                # Retrieve precomputed distances
                gradient_dist, texture_dist, center_dist, color_dist = all_distances[(i, j)]

                # Normalize each distance
                normalized_gradient_dist = normalize(gradient_dist, gradient_min, gradient_max)
                normalized_texture_dist = normalize(texture_dist, texture_min, texture_max)
                normalized_center_dist = normalize(center_dist, center_min, center_max)
                normalized_color_dist = normalize(color_dist, color_min, color_max)  # New color normalization

                # Compute total normalized distance
                total_distance = (
                    normalized_gradient_dist +
                    normalized_texture_dist +
                    normalized_center_dist +
                    normalized_color_dist  # Include color in total
                )
                distances.append(total_distance)
                valid_indices.append(j)

        # Select the nearest non-shadow region
        if valid_indices:
            nearest_index = np.argmin(distances)
            near_labels[i] = valid_indices[nearest_index]

    return near_labels

# def shadow_removal(img, segmented_img, label, near_labels, center_indices,i):
#     """
#     Perform shadow removal based on histogram matching and boundary smoothing.
#     """
#     near_labels = find_non_shadow_pair(img,label, segmented_img, center_indices)
#     # draw nearest region
#     near_img = draw_nearest_region_only_shadow(img, segmented_img, center_indices, near_labels,label)
#     cv2.imwrite("result/{}_near_img.jpg".format(i), near_img)
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
#     h_channel, s_channel, v_channel = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

#     for shadow_idx, lbl in enumerate(label):
#         if lbl == 0:  # Shadow region
#             # Find the nearest non-shadow region
#             j = near_labels[shadow_idx]
#             if label[j] == 1:  # Ensure it's a non-shadow region
#                 # Perform histogram matching for each channel
#                 print("Performing histogram matching for shadow region {} and non-shadow region {}".format(i, j))
#                 shadow_mask = (segmented_img == shadow_idx)
#                 non_shadow_mask = (segmented_img == j)
                
#                 h_channel[shadow_mask] = hist_match(h_channel[shadow_mask], h_channel[non_shadow_mask])
#                 s_channel[shadow_mask] = hist_match(s_channel[shadow_mask], s_channel[non_shadow_mask])
#                 v_channel[shadow_mask] = hist_match(v_channel[shadow_mask], v_channel[non_shadow_mask])

#     h_channel = np.clip(h_channel, 0, 180)
#     s_channel = np.clip(s_channel, 0, 255)
#     v_channel = np.clip(v_channel, 0, 255)

#     # Combine back the channels
#     hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2] = h_channel, s_channel, v_channel

#     # Convert back to BGR color space
#     result_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

#     # Smooth shadow boundaries
#     kernel = cv2.getGaussianKernel(ksize=3, sigma=3)  # Adjust sigma for less aggressive smoothing
#     smooth_result_img = cv2.filter2D(result_img, -1, kernel)

#     return smooth_result_img


def shadow_removal(img, segmented_img, label, near_labels, center_indices, i):
    """
    Perform shadow removal based on histogram matching and boundary smoothing.
    """
    near_labels = find_non_shadow_pair(img, label, segmented_img, center_indices)
    # draw nearest region
    near_img = draw_nearest_region_only_shadow(img, segmented_img, center_indices, near_labels, label)
    cv2.imwrite("result/Paired_{}".format(i), near_img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_channel, s_channel, v_channel = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    for shadow_idx, lbl in enumerate(label):
        if lbl == 0:  # Shadow region
            # Find the nearest non-shadow region
            j = near_labels[shadow_idx]
            if label[j] == 1:  # Ensure it's a non-shadow region
                # Perform histogram matching for each channel
                # print("Performing histogram matching for shadow region {} and non-shadow region {}".format(i, j))
                shadow_mask = (segmented_img == shadow_idx)
                non_shadow_mask = (segmented_img == j)
                
                h_channel[shadow_mask] = hist_match(h_channel[shadow_mask], h_channel[non_shadow_mask])
                s_channel[shadow_mask] = hist_match(s_channel[shadow_mask], s_channel[non_shadow_mask])
                v_channel[shadow_mask] = hist_match(v_channel[shadow_mask], v_channel[non_shadow_mask])

    h_channel = np.clip(h_channel, 0, 180)
    s_channel = np.clip(s_channel, 0, 255)
    v_channel = np.clip(v_channel, 0, 255)

    # Combine back the channels
    hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2] = h_channel, s_channel, v_channel

    # Convert back to BGR color space
    result_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Smooth shadow boundaries
    shadow_edges = cv2.Canny((segmented_img > 0).astype(np.uint8) * 255, 100, 200)
    dilated_edges = cv2.dilate(shadow_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge_mask = dilated_edges > 0

    smooth_result_img = result_img.copy()
    result_blur = cv2.GaussianBlur(result_img, (9, 9), sigmaX=3, sigmaY=3)

    # Apply blur only to edge areas
    smooth_result_img[edge_mask] = result_blur[edge_mask]

    return smooth_result_img


"""
Main function
"""
def main():

    # os.makedirs("result", exist_ok=True)
    os.makedirs("Detect", exist_ok=True)
    os.makedirs("Remove", exist_ok=True)
    all_files = os.listdir("data/SRD/shadow")
    jpg_files = [f for f in all_files if f.endswith('.jpg')]
    Remove_exist = os.listdir("Remove")

    for i, file in enumerate(jpg_files):
        if file not in Remove_exist:
            # 如果 file _ 後面數字是6的話，就是我們要的
            if file[4] == '7':
                print("Processing image", file)
                img = cv2.imread(os.path.join("data/SRD/shadow", file))

                # TODO: STEP1 SHADOW DETECTION

                # segmented_img, border_img = mean_shift(img)
                segmented_img, border_img = mean_shift_with_merge(img, min_region_size=500)
                # segmented_img, border_img = quick_shift(img)   

                #cv2.imshow("meanshift", border_img)
                #cv2.waitKey(0)
                #cv2.imwrite("result/input{}_meanshift.jpg".format(i + 1), border_img)
                
                # np.save('segmented_img_input{}.npy'.format(i), segmented_img)
                # segmented_img = np.load('segmented_img_input{}.npy'.format(i))

                center_indices, center_marked_img = find_center(img, segmented_img)

                #cv2.imshow("center_indices", center_marked_img)
                #cv2.waitKey(0)
                #cv2.imwrite("result/input{}_center_marked.jpg".format(i + 1), center_marked_img)

                near_labels, grad_mag, lbp_norm = find_nearest_region(img, segmented_img, center_indices)

                #cv2.imshow("gradient_img".format(i+1), grad_mag.astype(np.uint8))
                #cv2.waitKey(0)
                #cv2.imwrite("result/input{}_gradient.jpg".format(i+1), grad_mag)
                
                #cv2.imshow("texture_img", lbp_norm)
                #cv2.waitKey(0)
                #cv2.imwrite("result/input{}_texture.jpg".format(i+1), lbp_norm)

                near_img = draw_nearest_region(img, segmented_img, center_indices, near_labels)

                # cv2.imshow("near_img".format(i+1), near_img)
                # cv2.waitKey(0)
                # cv2.imwrite("result/{}_nearest_region.jpg".format(i), near_img)

                cluster_centers, cluster_std, kmeans_img = shadow_light_cluster(img, segmented_img)

                # cv2.imshow("kmeans", kmeans_img)
                # cv2.waitKey(0)
                # cv2.imwrite("result/input{}_kmeans.jpg".format(i + 1), kmeans_img)

                label, shadow_detect_img= shadow_detection(img, segmented_img, cluster_centers, cluster_std, near_labels)
                
                # cv2.imshow("shadow_detect", shadow_detect_img)
                # cv2.waitKey(0)
                cv2.imwrite("Detect/{}".format(file), shadow_detect_img)


                # TODO: STEP2 SHADOW REMOVAL
                remove_img = shadow_removal(img, segmented_img, label, near_labels, center_indices,file)

                # cv2.imshow("remove_img", remove_img)
                # cv2.waitKey(0)
                cv2.imwrite("Remove/{}".format(file), remove_img)
            


if __name__ == "__main__":
    main()

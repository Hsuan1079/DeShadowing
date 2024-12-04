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
        kernel_size=9,    # 對應 MATLAB 的 SpatialBandWidth
        max_dist=15,      # 對應 MATLAB 的 RangeBandWidth
        ratio=0.1         # 可調參數，用於控制區域間距離
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
    bandwidth = estimate_bandwidth(flat_img_with_coordinates, quantile=0.006, n_samples=500)

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


def cal_center_dist(img, i, j, center_indices):
    height, width, _ = img.shape
    dist = np.sum((center_indices[i] - center_indices[j]) ** 2) ** 0.5
    dist_norm = dist / max(height, width)
    #print("dist_norm[{}, {}]".format(i, j), dist_norm)
    return dist_norm

def find_nearest_region(img, segmented_img, center_indices):
    label_num = len(np.unique(segmented_img))
    near_labels = np.full((label_num), -1)
    print("near_labels.shape", near_labels.shape)

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
    print("near_labels", near_labels)

    return near_labels, grad_mag, lbp_norm

def draw_nearest_region(img, segmented_img, center_indices, near_labels):
    label_num = len(np.unique(segmented_img))
    near_img = (mark_boundaries(img, segmented_img, color=(0, 0, 0)) * 255).astype(np.uint8)
    for i in range(label_num):
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

    print("np.hstack(R.ravel(), c)")
    print(np.transpose(np.vstack([R.ravel(), cluster_labels])))
    print("cluster_centers =", cluster_centers)
    print("cluster_std =", cluster_std)

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
    print("Y_mean", Y_mean)

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
        print("max_F_shadow =", max_F_shadow)            
        print("max_F_shadow_label =", max_F_shadow_label)
        if(not update): #if update == False || max_F_shadow < 0.0028
            print("no update: break")
            break
        print("update")
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


"""
Main function
"""
def main():

    os.makedirs("result", exist_ok=True)

    for i in range(9):
        print("input{}.jpg".format(i + 1))
        img = cv2.imread("data/input{}.jpg".format(i + 1))

        segmented_img, border_img = mean_shift(img)
        #segmented_img, border_img = quick_shift(img)

        #cv2.imshow("meanshift", border_img)
        #cv2.waitKey(0)
        #cv2.imwrite("result/input{}_meanshift.jpg".format(i + 1), border_img)
        
        np.save('segmented_img_input{}.npy'.format(i + 1), segmented_img)
        segmented_img = np.load('segmented_img_input{}.npy'.format(i + 1))

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

        #cv2.imshow("near_img".format(i+1), near_img)
        #cv2.waitKey(0)
        #cv2.imwrite("result/input{}_nearest_region.jpg".format(i+1), near_img)

        cluster_centers, cluster_std, kmeans_img = shadow_light_cluster(img, segmented_img)

        cv2.imshow("kmeans", kmeans_img)
        cv2.waitKey(0)
        cv2.imwrite("result/input{}_kmeans.jpg".format(i + 1), kmeans_img)

        label, shadow_detect_img= shadow_detection(img, segmented_img, cluster_centers, cluster_std, near_labels)
        
        cv2.imshow("shadow_detect", shadow_detect_img)
        cv2.waitKey(0)
        cv2.imwrite("result/input{}_shadow_detect.jpg".format(i + 1), shadow_detect_img)


if __name__ == "__main__":
    main()
# only change histogram
import os
import cv2
import numpy as np


def remove_small_regions(mask, min_region_size=500):
    """
    移除小於 min_region_size 的連通區域。
    """
    # 找到連通區域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 保留大區域
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):  # 忽略背景標籤 0
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            cleaned_mask[labels == i] = 255
    return cleaned_mask

def match_histograms(source, reference):
    source_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    source_cdf = np.cumsum(source_hist).astype(float)
    reference_cdf = np.cumsum(reference_hist).astype(float)

    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        closest_idx = np.argmin(np.abs(reference_cdf - source_cdf[i]))
        lookup_table[i] = closest_idx

    matched = cv2.LUT(source, lookup_table)
    return matched

def enhance_shadow_with_histogram_matching(image_path, blur_kernel_size=15, min_region_size=500):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("無法讀取影像，請確認路徑是否正確")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 陰影檢測
    _, shadow_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 清理小區塊
    shadow_mask = remove_small_regions(shadow_mask, min_region_size)

    # 平滑遮罩
    shadow_mask_blurred = cv2.GaussianBlur(shadow_mask, (blur_kernel_size, blur_kernel_size), 0)
    shadow_mask_normalized = shadow_mask_blurred / 255.0
    
    non_shadow_mask = cv2.bitwise_not(shadow_mask)

    # 非陰影區域
    h_non_shadow = cv2.bitwise_and(h, h, mask=non_shadow_mask)
    s_non_shadow = cv2.bitwise_and(s, s, mask=non_shadow_mask)
    v_non_shadow = cv2.bitwise_and(v, v, mask=non_shadow_mask)

    # 陰影區域
    h_shadow = cv2.bitwise_and(h, h, mask=shadow_mask)
    s_shadow = cv2.bitwise_and(s, s, mask=shadow_mask)
    v_shadow = cv2.bitwise_and(v, v, mask=shadow_mask)

    # 直方圖匹配
    h_shadow_adjusted = match_histograms(h_shadow, h_non_shadow)
    s_shadow_adjusted = match_histograms(s_shadow, s_non_shadow)
    v_shadow_adjusted = match_histograms(v_shadow, v_non_shadow)

    # 合併
    h_combined = np.where(shadow_mask > 0, h_shadow_adjusted, h)
    s_combined = np.where(shadow_mask > 0, s_shadow_adjusted, s)
    v_combined = np.where(shadow_mask > 0, v_shadow_adjusted, v)

    hsv_adjusted = cv2.merge([h_combined, s_combined, v_combined])
    image_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    # 顯示結果
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Cleaned Shadow Mask", shadow_mask)
    # cv2.imshow("Enhanced Image (Shadow Adjusted)", image_adjusted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_adjusted, shadow_mask

# image_path = '1.jpg'  # 替換為你的影像檔案路徑
# brightness_increase = 70  # 設定亮度增加的數值
# blur_kernel_size = 15  # 平滑遮罩的高斯核大小
# enhanced_image, shadow_mask = enhance_shadow_with_histogram_matching(image_path)
# cv2.imwrite('output{}'.format(image_path), enhanced_image)

# 呼叫函式並傳入影像路徑
all_files = os.listdir("data")
jpg_files = [f for f in all_files if f.endswith('.jpg')]

output_dirs = ['result_histogram/mask', 'result_histogram/enhanced']
for directory in output_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"創建資料夾: {directory}")
        
for i, file in enumerate(jpg_files):
    print("Processing image", file)
    image_path = os.path.join("data", file)
    enhanced_image, shadow_mask = enhance_shadow_with_histogram_matching(image_path)    
    # 儲存結果
    cv2.imwrite('result_histogram/mask/' + file, shadow_mask)
    cv2.imwrite('result_histogram/enhanced/' + file, enhanced_image)
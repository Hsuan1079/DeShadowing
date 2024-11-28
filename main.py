import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect(image):
    return NotImplementedError

def split_mask_by_fixed_colors(image_path, shadow_mask):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请确认路径是否正确")

    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 提取阴影区域
    shadow_region = shadow_mask > 0  # 阴影区域为非零部分
    h_shadow = h[shadow_region]
    s_shadow = s[shadow_region]

    # 定义颜色区间（单位：H 通道值 0-180）
    color_ranges = {
        "Red": [(0, 15), (165, 180)],
        "Orange": [(15, 30)],
        "Yellow": [(30, 60)],
        "Green": [(60, 120)],
        "Blue": [(120, 135)],
        "Indigo": [(135, 150)],
        "Purple": [(150, 165)],
        "Other": []  # 未分组的颜色
    }

    # 初始化遮罩分类
    masks = {}
    for color in color_ranges:
        masks[color] = np.zeros_like(shadow_mask, dtype=np.uint8)

    # 分配颜色到不同类别
    for color, ranges in color_ranges.items():
        for r in ranges:
            in_range = (h_shadow >= r[0]) & (h_shadow < r[1])
            masks[color][shadow_region] |= in_range.astype(np.uint8) * 255

    # 统计非空类别并返回
    valid_masks = {color: mask for color, mask in masks.items() if np.any(mask > 0)}

    # 显示分组后的遮罩
    for color, mask in valid_masks.items():
        cv2.imshow(f"Mask for {color}", mask)
        cv2.imwrite(f"mask_{color}.png", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return valid_masks


def split_mask_by_color(image_path, shadow_mask):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请确认路径是否正确")

    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 提取阴影区域
    shadow_region = shadow_mask > 0  # 阴影区域为非零部分

    # 获取阴影区域的颜色信息 (例如 H 和 S)
    h_shadow = h[shadow_region]
    s_shadow = s[shadow_region]

    # 使用颜色聚类 (K-Means) 将阴影区域分为多个子区域
    color_features = np.column_stack((h_shadow, s_shadow))  # 使用色调和饱和度作为聚类特征
    # num_cluster 根據有的顏色數量來調整，顏色相近的當作同一個cluster
    num_clusters = 3  # 聚类数量
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(color_features)
    labels = kmeans.labels_  # 聚类标签

    # 创建多个子遮罩
    masks = []
    for i in range(num_clusters):
        # 根据聚类标签生成子遮罩
        sub_mask = np.zeros_like(shadow_mask, dtype=np.uint8)
        sub_mask[shadow_region] = (labels == i).astype(np.uint8) * 255  # 聚类标签对应的区域
        masks.append(sub_mask)
    
    # 把三個mask 顯示出來
    for i in range(num_clusters):
        cv2.imshow(f"Mask {i}", masks[i])
    cv2.waitKey(0)
    return masks

def enhance_shadow_brightness(image_path, brightness_increase=50, blur_kernel_size=15):
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("無法讀取影像，請確認路徑是否正確")

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 使用大津法自動閾值分割亮度 (Value)
    _, shadow_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # shadow_mask = detect(image_path)


    # 平滑遮罩邊緣
    shadow_mask_blurred = cv2.GaussianBlur(shadow_mask, (blur_kernel_size, blur_kernel_size), 0)
    shadow_mask_normalized = shadow_mask_blurred / 255.0  # 正規化到 [0, 1]

    masks = split_mask_by_color(image_path, shadow_mask_normalized)
    # masks = split_mask_by_fixed_colors(image_path, shadow_mask_normalized)
    exit()

    # 找到遮罩的非零位置
    shadow_indices = np.argwhere(shadow_mask_normalized > 0.95)

    # 计算非阴影区域的平均亮度作为参考亮度
    non_shadow_mask = 1 - shadow_mask_normalized  # 非阴影区域
    reference_brightness = np.mean(v[non_shadow_mask > 0])

    for x, y in shadow_indices:
        current_brightness = v[x, y]  # 当前像素亮度
        brightness_ratio = reference_brightness / max(current_brightness, 1)  # 避免除以 0
        adjusted_brightness = np.clip(current_brightness * (brightness_ratio/0.75), 0, 255)  # 调整亮度
        v[x, y] = adjusted_brightness  # 更新亮度通道
        print(f"Pixel ({x}, {y}): {current_brightness} -> {adjusted_brightness}")
        
    # 将修改后的 V 通道合并回 HSV，并转换回 BGR
    hsv_corrected = cv2.merge([h, s, v])
    corrected_image = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    # 顯示結果
    cv2.imshow("Original Image", image)
    cv2.imshow("Shadow Mask (Original)", shadow_mask)
    # cv2.imshow("Shadow Mask (Blurred)", shadow_mask_blurred)
    cv2.imshow("Enhanced Image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corrected_image, shadow_mask

# 呼叫函式並傳入影像路徑
image_path = 'data/food2.jpg'  # 替換為你的影像檔案路徑
brightness_increase = 60  # 設定亮度增加的數值
blur_kernel_size = 15  # 平滑遮罩的高斯核大小
enhanced_image, shadow_mask = enhance_shadow_brightness(image_path, brightness_increase, blur_kernel_size)
cv2.imwrite('output.jpeg', enhanced_image)
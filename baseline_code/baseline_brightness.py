# only change brightness
import os
import cv2
import numpy as np

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

    # 平滑遮罩邊緣
    shadow_mask_blurred = cv2.GaussianBlur(shadow_mask, (blur_kernel_size, blur_kernel_size), 0)
    shadow_mask_normalized = shadow_mask_blurred / 255.0  # 正規化到 [0, 1]

    # 提高影子區域的亮度，並應用平滑過渡
    v_enhanced = v.copy()
    adjusted_brightness = np.clip(v_enhanced + (brightness_increase * shadow_mask_normalized), 0, 255).astype(np.uint8)

    # 合併回 HSV 並轉換回 BGR
    hsv_enhanced = cv2.merge([h, s, adjusted_brightness])
    image_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # 顯示結果
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Shadow Mask (Original)", shadow_mask)
    # cv2.imshow("Shadow Mask (Blurred)", shadow_mask_blurred)
    # cv2.imshow("Enhanced Image", image_enhanced)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_enhanced, shadow_mask

# 呼叫函式並傳入影像路徑
# image_path = '1.jpg'  # 替換為你的影像檔案路徑
# enhanced_image, shadow_mask = enhance_shadow_brightness(image_path, brightness_increase, blur_kernel_size)
# cv2.imwrite('output{}'.format(image_path), enhanced_image)
brightness_increase = 70  # 設定亮度增加的數值
blur_kernel_size = 15  # 平滑遮罩的高斯核大小
all_files = os.listdir("data")
jpg_files = [f for f in all_files if f.endswith('.jpg')]

output_dirs = ['result_brightness/mask', 'result_brightness/enhanced']
for directory in output_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"創建資料夾: {directory}")
        
for i, file in enumerate(jpg_files):
    print("Processing image", file)
    image_path = os.path.join("data", file)
    enhanced_image, shadow_mask = enhance_shadow_brightness(image_path, brightness_increase, blur_kernel_size)   
    # 儲存結果
    cv2.imwrite('result_brightness/mask/' + file, shadow_mask)
    cv2.imwrite('result_brightness/enhanced/' + file, enhanced_image)
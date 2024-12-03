import cv2
import numpy as np
from segment_and_extract_features import calculate_features, segment,segment_image, edison_meanshift_segmentation
from skimage import color


def main():
    # Step 1: 讀取輸入圖像
    image_path = "data/4.jpg"  
    image = cv2.imread(image_path)
    if image is None:
        print("圖像加載失敗，請檢查路徑。")
        return

    # Step 2: segementation
    print("segmenting...")
    seg, segnum = segment(image)
    # seg, segnum = mean_shift_segment(image)
    # seg, segnum = segment_image(image)
    seg, segnum = edison_meanshift_segmentation(image)
    
    # 顯示分割結果
    print(f"分割完成，區域數量: {segnum}")
    # seg_vis = color.label2rgb(seg, image, kind='avg')  # 將區域顯示為平均顏色
    # cv2.imshow("Segmented Image", seg_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 3: extract features
    print("extracting features...")
    features = calculate_features(image, seg,segnum)



    # # Step 3: 區域匹配
    # matching_pairs = compute_region_matches(features)
    # print(f"完成區域匹配，共找到 {len(matching_pairs)} 對區域。")

    # # Step 4: 陰影檢測
    # shadow_labels = detect_shadows(features)
    # print(f"完成初步陰影檢測。陰影區域數量：{np.sum(shadow_labels == 1)}")

    # # Step 5: 陰影標籤校正
    # corrected_labels = refine_shadow_labels(features, shadow_labels, matching_pairs)
    # print("完成陰影標籤校正。")

    # # Step 6: 陰影去除
    # shadow_removed_image = remove_shadows(image, regions, corrected_labels)
    # print("完成陰影去除。")

    # # Step 7: 保存和顯示結果
    # output_image_path = "path/to/output/image.jpg"
    # cv2.imwrite(output_image_path, shadow_removed_image)
    # print(f"陰影去除結果已保存至 {output_image_path}")

    # # 選擇性顯示結果
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Shadow Removed Image", shadow_removed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import peak_signal_noise_ratio

remove_files = os.listdir("output/Remove")
baseline_pairing = os.listdir("baseline_result/baseline_pairing")
baseline_brightness = os.listdir("baseline_result/baseline_brightness/enhanced")
baseline_histogram = os.listdir("baseline_result/baseline_histogram/enhanced")

# 儲存 PSNR 結果
results = []
results_pairing = []
results_brightness = []
results_histogram = []

for i in remove_files:
    # 載入影像
    if i == ".DS_Store":
        continue
    gdname = i.split(".")[0]+"_free.jpg"
    print(gdname)
    print(i)
    recovered_image = cv2.imread("output/Remove/"+i)
    baseline_pairing = cv2.imread("baseline_result/baseline_pairing/"+i)
    baseline_brightness = cv2.imread("baseline_result/baseline_brightness/enhanced/"+i)
    baseline_histogram = cv2.imread("baseline_result/baseline_histogram/enhanced/"+i)
    ground_truth_image = cv2.imread("data/shadow_free/"+gdname)

    
    # 計算 PSNR
    value = peak_signal_noise_ratio(recovered_image, ground_truth_image)
    value_baseline_pairing = peak_signal_noise_ratio(baseline_pairing, ground_truth_image)
    value_baseline_brightness = peak_signal_noise_ratio(baseline_brightness, ground_truth_image)
    value_baseline_histogram = peak_signal_noise_ratio(baseline_histogram, ground_truth_image)
    
    # 儲存結果
    results.append({"recovered_image": i, "psnr": value})
    results_pairing.append({"baseline": i, "psnr": value_baseline_pairing})
    results_brightness.append({"baseline": i, "psnr": value_baseline_brightness})
    results_histogram.append({"baseline": i, "psnr": value_baseline_histogram})

# 將結果轉換為 DataFrame 兩格存到一起
results_df = pd.DataFrame(results)
results_df_pairing = pd.DataFrame(results_pairing)
results_df_brightness = pd.DataFrame(results_brightness)
results_df_histogram = pd.DataFrame(results_histogram)
results_df = pd.concat([results_df, results_df_pairing, results_df_brightness, results_df_histogram], axis=1)
results_df.columns = ["recovered_image", "psnr", "pairing", "psnr_pairing", "brightness", "psnr_brightness", "histogram", "psnr_histogram"]

# 計算平均 psnr
average_psnr = results_df["psnr"].mean()
average_psnr_pairing = results_df["psnr_pairing"].mean()
average_psnr_brightness = results_df["psnr_brightness"].mean()
average_psnr_histogram = results_df["psnr_histogram"].mean()
print(f"平均 psnr: {average_psnr}")
print(f"平均 psnr_baseline 1-pairing: {average_psnr_pairing}")
print(f"平均 psnr_baseline 2-brightness: {average_psnr_brightness}")
print(f"平均 psnr_baseline 3- histogram: {average_psnr_histogram}")

# 計算最佳與最差的 psnr
min_psnr = results_df["psnr"].min()
max_psnr = results_df["psnr"].max()
min_psnr_pairing = results_df["psnr_pairing"].min()
max_psnr_pairing = results_df["psnr_pairing"].max()
min_psnr_brightness = results_df["psnr_brightness"].min()
max_psnr_brightness = results_df["psnr_brightness"].max()
min_psnr_histogram = results_df["psnr_histogram"].min()
max_psnr_histogram = results_df["psnr_histogram"].max()
print(f"最佳 psnr: {min_psnr}")
print(f"最差 psnr: {max_psnr}")
print(f"最佳 psnr_baseline: {min_psnr_pairing}")
print(f"最差 psnr_baseline: {max_psnr_pairing}")
print(f"最佳 psnr_baseline: {min_psnr_brightness}")
print(f"最差 psnr_baseline: {max_psnr_brightness}")
print(f"最佳 psnr_baseline: {min_psnr_histogram}")
print(f"最差 psnr_baseline: {max_psnr_histogram}")

# 結果也存進去 csv 寫在最下面那行
new_df = pd.DataFrame([{"average_psnr": average_psnr, "average_psnr_pairing": average_psnr_pairing, 
                        "average_psnr_brightness": average_psnr_brightness, "average_psnr_histogram": average_psnr_histogram, 
                        "min_psnr": min_psnr, 
                        "max_psnr": max_psnr, 
                        "min_psnr_pairing": min_psnr_pairing, 
                        "max_psnr_pairing": max_psnr_pairing,
                        "min_psnr_brightness": min_psnr_brightness, 
                        "max_psnr_brightness": max_psnr_brightness,
                        "min_psnr_histogram": min_psnr_histogram, 
                        "max_psnr_histogram": max_psnr_histogram
                        }])
results_df = pd.concat([results_df, new_df], axis=1)



# 將結果存為 CSV 檔案
results_df.to_csv("psnr_results.csv", index=False, encoding="utf-8")
print("已將結果存入 psnr_results.csv")

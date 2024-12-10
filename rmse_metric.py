import cv2
import numpy as np
import pandas as pd
import os

def calculate_rmse(image1, image2):

    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab)
    rmse = np.sqrt(np.mean((lab1 - lab2) ** 2))
    return rmse


remove_files = os.listdir("output/Remove")
baseline_pairing = os.listdir("baseline_result/baseline_pairing")
baseline_brightness = os.listdir("baseline_result/baseline_brightness/enhanced")
baseline_histogram = os.listdir("baseline_result/baseline_histogram/enhanced")

# 儲存 RMSE 結果
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

    
    # 計算 RMSE
    rmse_value = calculate_rmse(recovered_image, ground_truth_image)
    rmse_value_baseline_pairing = calculate_rmse(baseline_pairing, ground_truth_image)
    rmse_value_baseline_brightness = calculate_rmse(baseline_brightness, ground_truth_image)
    rmse_value_baseline_histogram = calculate_rmse(baseline_histogram, ground_truth_image)
    
    # 儲存結果
    results.append({"recovered_image": i, "rmse": rmse_value})
    results_pairing.append({"baseline": i, "rmse": rmse_value_baseline_pairing})
    results_brightness.append({"baseline": i, "rmse": rmse_value_baseline_brightness})
    results_histogram.append({"baseline": i, "rmse": rmse_value_baseline_histogram})

# 將結果轉換為 DataFrame 兩格存到一起
results_df = pd.DataFrame(results)
results_df_pairing = pd.DataFrame(results_pairing)
results_df_brightness = pd.DataFrame(results_brightness)
results_df_histogram = pd.DataFrame(results_histogram)
results_df = pd.concat([results_df, results_df_pairing, results_df_brightness, results_df_histogram], axis=1)
results_df.columns = ["recovered_image", "rmse", "pairing", "rmse_pairing", "brightness", "rmse_brightness", "histogram", "rmse_histogram"]

# 計算平均 RMSE
average_rmse = results_df["rmse"].mean()
average_rmse_pairing = results_df["rmse_pairing"].mean()
average_rmse_brightness = results_df["rmse_brightness"].mean()
average_rmse_histogram = results_df["rmse_histogram"].mean()
print(f"平均 RMSE: {average_rmse}")
print(f"平均 RMSE_baseline 1-pairing: {average_rmse_pairing}")
print(f"平均 RMSE_baseline 2-brightness: {average_rmse_brightness}")
print(f"平均 RMSE_baseline 3- histogram: {average_rmse_histogram}")

# 計算最佳與最差的 RMSE
min_rmse = results_df["rmse"].min()
max_rmse = results_df["rmse"].max()
min_rmse_pairing = results_df["rmse_pairing"].min()
max_rmse_pairing = results_df["rmse_pairing"].max()
min_rmse_brightness = results_df["rmse_brightness"].min()
max_rmse_brightness = results_df["rmse_brightness"].max()
min_rmse_histogram = results_df["rmse_histogram"].min()
max_rmse_histogram = results_df["rmse_histogram"].max()
print(f"最佳 RMSE: {min_rmse}")
print(f"最差 RMSE: {max_rmse}")
print(f"最佳 RMSE_baseline: {min_rmse_pairing}")
print(f"最差 RMSE_baseline: {max_rmse_pairing}")
print(f"最佳 RMSE_baseline: {min_rmse_brightness}")
print(f"最差 RMSE_baseline: {max_rmse_brightness}")
print(f"最佳 RMSE_baseline: {min_rmse_histogram}")
print(f"最差 RMSE_baseline: {max_rmse_histogram}")

# 結果也存進去 csv 寫在最下面那行
new_df = pd.DataFrame([{"average_rmse": average_rmse, "average_rmse_pairing": average_rmse_pairing, 
                        "average_rmse_brightness": average_rmse_brightness, "average_rmse_histogram": average_rmse_histogram, 
                        "min_rmse": min_rmse, 
                        "max_rmse": max_rmse, 
                        "min_rmse_pairing": min_rmse_pairing, 
                        "max_rmse_pairing": max_rmse_pairing,
                        "min_rmse_brightness": min_rmse_brightness, 
                        "max_rmse_brightness": max_rmse_brightness,
                        "min_rmse_histogram": min_rmse_histogram, 
                        "max_rmse_histogram": max_rmse_histogram
                        }])
results_df = pd.concat([results_df, new_df], axis=1)



# 將結果存為 CSV 檔案
results_df.to_csv("rmse_results.csv", index=False, encoding="utf-8")
print("已將結果存入 rmse_results.csv")

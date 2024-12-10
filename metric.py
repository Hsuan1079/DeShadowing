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
baseliens = os.listdir("baseline")

# 儲存 RMSE 結果
results = []
results_baseline = []

for i in remove_files:
    # 載入影像
    if i == ".DS_Store":
        continue
    gdname = i.split(".")[0]+"_free.jpg"
    print(gdname)
    print(i)
    recovered_image = cv2.imread("output/Remove/"+i)
    baseliens = cv2.imread("baseline/"+i)
    ground_truth_image = cv2.imread("data/shadow_free/"+gdname)

    
    # 計算 RMSE
    rmse_value = calculate_rmse(recovered_image, ground_truth_image)
    rmse_value_baseline = calculate_rmse(baseliens, ground_truth_image)
    
    # 儲存結果
    results.append({"recovered_image": i, "rmse": rmse_value})
    results_baseline.append({"baseline": i, "rmse": rmse_value_baseline})

# 將結果轉換為 DataFrame 兩格存到一起
results_df = pd.DataFrame(results)
results_df_baseline = pd.DataFrame(results_baseline)
results_df = pd.concat([results_df, results_df_baseline], axis=1)
results_df.columns = ["recovered_image", "rmse", "baseline", "rmse_baseline"]

# 計算平均 RMSE
average_rmse = results_df["rmse"].mean()
average_rmse_baseline = results_df["rmse_baseline"].mean()
print(f"平均 RMSE: {average_rmse}")
print(f"平均 RMSE_baseline: {average_rmse_baseline}")

# 計算最佳與最差的 RMSE
min_rmse = results_df["rmse"].min()
max_rmse = results_df["rmse"].max()
min_rmse_baseline = results_df["rmse_baseline"].min()
max_rmse_baseline = results_df["rmse_baseline"].max()
print(f"最佳 RMSE: {min_rmse}")
print(f"最差 RMSE: {max_rmse}")
print(f"最佳 RMSE_baseline: {min_rmse_baseline}")
print(f"最差 RMSE_baseline: {max_rmse_baseline}")

# 結果也存進去 csv 寫在最下面那行
new_df = pd.DataFrame([{"average_rmse": average_rmse, "average_rmse_baseline": average_rmse_baseline, "min_rmse": min_rmse, "max_rmse": max_rmse, "min_rmse_baseline": min_rmse_baseline, "max_rmse_baseline": max_rmse_baseline}])
results_df = pd.concat([results_df, new_df], axis=1)



# 將結果存為 CSV 檔案
results_df.to_csv("rmse_results.csv", index=False, encoding="utf-8")
print("已將結果存入 rmse_results.csv")

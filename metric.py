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

# 儲存 RMSE 結果
results = []

for i in remove_files:
    # 載入影像
    if i == ".DS_Store":
        continue
    gdname = i.split(".")[0]+"_free.jpg"
    print(gdname)
    print(i)
    recovered_image = cv2.imread("output/Remove/"+i)
    ground_truth_image = cv2.imread("data/shadow_free/"+gdname)

    
    # 計算 RMSE
    rmse_value = calculate_rmse(recovered_image, ground_truth_image)
    
    # 儲存結果
    results.append({"recovered_image": i, "rmse": rmse_value})

# 將結果轉換為 DataFrame
results_df = pd.DataFrame(results)

# 將結果存為 CSV 檔案
results_df.to_csv("ours_rmse_results.csv", index=False, encoding="utf-8")
print("已將結果存入 rmse_results.csv")

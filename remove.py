import os
import cv2

os.makedirs("output/Remove", exist_ok=True)
os.makedirs("output/new", exist_ok=True)
os.makedirs("Data_choose", exist_ok=True)
all_files = os.listdir("output/Paired")
remove = os.listdir("output/Remove")
remove_t =[i.split(".")[0] for i in remove]
remove_t = remove_t[1:]
print(remove_t)

# 檢查all_files裡面的檔案是否在remove裡面，如果有就移動到Data_choose資料夾
for i in all_files:
    word = i.split(".")
    if word[0][7:] in remove_t:
        # 把資料寫進new資料夾
        img = cv2.imread("output/Paired/"+i)
        cv2.imwrite("output/new/"+i, img)
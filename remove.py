import os
import cv2

os.makedirs("Detect", exist_ok=True)
os.makedirs("Remove", exist_ok=True)
os.makedirs("Data_choose", exist_ok=True)
all_files = os.listdir("data/SRD/shadow_free")
remove = os.listdir("Remove")
remove_t =[i.split(".")[0] for i in remove]
remove_t = remove_t[1:]
print(remove_t)

# 檢查all_files裡面的檔案是否在remove裡面，如果有就移動到Data_choose資料夾
for i in all_files:
    word = i.split(".")
    if word[0][:-5] in remove_t:
        cv2.imwrite("data/Data_choose_shadow_free/" + i, cv2.imread("data/SRD/shadow_free/" + i))

import cv2 
import os

dir="lp_crop/crop"
list_item=os.listdir(dir)

if not os.path.isdir("lp_crop/resize"):
    os.mkdir("lp_crop/resize")

for i in list_item:
    array=cv2.imread(os.path.join(dir,i))
    resize_array=cv2.resize(array,(128,128))

    cv2.imwrite(f"lp_crop/resize/resize_{i}",resize_array)
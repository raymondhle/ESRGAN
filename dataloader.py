import torch
from tqdm import tqdm
import time
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
from PIL import Image
import cv2


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

        self.data=sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        root_and_dir = os.path.join(self.root_dir,img_file)

        image = cv2.imread(root_and_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        both_transform = config.both_transforms(image=image)["image"]
        low_res = config.lowres_transform(image=both_transform)["image"]
        high_res = config.highres_transform(image=both_transform)["image"]
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="lp_crop/resize")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}

mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestDataset(Dataset):
    def __init__(self, root_dir, transform, mode) -> None:
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode

        if self.mode == 'train':
            label_file = 'train_val_list.txt'
        else:
            label_file = 'test_list.txt'

        gr_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        gr = pd.read_csv(gr_path, index_col=0)
        gr = gr.to_dict()["Finding Labels"]

        img_list = os.path.join(root_dir, label_file)
        with open(img_list) as f:
            all_names = f.read().splitlines()

        self.all_imgs = np.asarray([x for x in all_names])
        self.gr_str = np.asarray([gr[i] for i in self.all_imgs])
        self.gr = np.zeros((self.gr_str.shape[0], 15))
        for idx, i in enumerate(self.gr_str):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.gr[idx] = binary_result

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        img_w = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return img_w, target, img_path
        # return img_w, target, self.gr_str[index]


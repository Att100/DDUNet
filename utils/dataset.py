import paddle
from paddle.io import Dataset
from PIL import Image
import pandas as pd
import random
import os
import numpy as np

RANDOM_SEED = 999

def read_metadata(path, type='all', split='train') -> list:
    random.seed(RANDOM_SEED)
    data = pd.read_csv(os.path.join(path, 'metadata.csv'))
    names = data['Name'].to_list()
    augmented = []
    for name in names:
        if type != 'all':
            if type == 'd':
                if name[0] != 'd':
                    continue
            elif type == 'n':
                if name[0] != 'n':
                    continue
        augmented.append(name)
        for i in range(1, 5+1):
            augmented.append(name.split(".")[0]+"_"+str(i)+".jpg")
    random.shuffle(augmented)
    length = len(augmented)
    if split == 'train':
        return augmented[:int(length*0.8)]
    else:
        return augmented[int(length*0.8):]

# class SWINySEG(Dataset):
#     def __init__(self, path="./dataset/SWINySEG", type='all', split='train'):
#         super().__init__()

#         self.path = path
#         self.split = split
#         self.names = read_metadata(path, type, split)

#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.path, 'images', self.names[idx]))
#         gt = Image.open(os.path.join(
#             self.path, 'GTmaps', self.names[idx].split(".")[0]+".png"))
#         img = img.resize((304, 304))
#         gt = gt if self.split=='test' else gt.resize((304, 304))
#         # to numpy array and normalize
#         img_arr = np.array(img).transpose(2, 0, 1) / 255
#         gt_arr = np.array(gt) / 255
#         img_tensor = paddle.to_tensor(img_arr).astype('float32')
#         gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
#         return img_tensor, gt_tensor

#     def __len__(self):
#         return len(self.names)


class SWINySEG(Dataset):
    def __init__(self, path="./dataset/SWINySEG", daynight='all', split='train', img_size=(304, 304), aug=True):
        super().__init__()

        self.path = path
        self.split = split
        self.aug = aug
        self.img_size = img_size
        self.names = [line.strip() for line in open(os.path.join(path, split+".txt")).readlines()]
        
        if daynight == 'd':
            self.names = [name for name in self.names if name[0] == 'd']
        elif daynight == 'n':
            self.names = [name for name in self.names if name[0] == 'n']
        else:
            pass

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, 'images', self.names[idx]+".jpg"))
        gt = Image.open(os.path.join(self.path, 'GTmaps', self.names[idx]+".png"))
        img = img.resize(self.img_size)
        gt = gt.resize(self.img_size) if self.split=='train' else gt
        # to numpy array and normalize
        img_arr = np.array(img, dtype='float32').transpose(2, 0, 1) / 255
        gt_arr = np.array(gt, dtype='float32') / 255
        img_arr = (img_arr - 0.5) / 0.5
        if self.split == 'train' and self.aug:
            # random h flip
            choice = np.random.choice([0, 1])
            if choice == 1:
                img_arr = img_arr[:, :, ::-1]
                gt_arr = gt_arr[:, ::-1]
            # random v flip
            choice = np.random.choice([0, 1])
            if choice == 1:
                img_arr = img_arr[:, ::-1, :]
                gt_arr = gt_arr[::-1, :]
        img_tensor = paddle.to_tensor(img_arr)
        gt_tensor = paddle.to_tensor(gt_arr)
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.names)
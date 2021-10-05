import paddle
import os
import numpy as np
from PIL import Image
import cv2


class CPDataset(paddle.io.Dataset):
    def __init__(self, data_txt, mode='train', num_classes=150, transforms=None, ignore_index=255, start_index_with_1=True):
        super(CPDataset, self).__init__()
        assert mode in ['train', 'val', 'test'], "mode is one of ['train', 'val', 'test']"
        self.mode = mode
        self.num_classes = num_classes
        self.transforms = transforms
        self.ignore_index = ignore_index
        self.start_index_with_1 = start_index_with_1
        self.data = []
        with open(data_txt) as f:
            for line in f.readlines():
                if mode != 'test':
                    p1, p2 = line.strip().split(' ')
                    self.data.append([p1, p2])
                else:
                    self.data.append(os.path.join(line.strip()))
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            img = cv2.imread(self.data[idx][0]).astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = np.array(Image.open(self.data[idx][1]))
            if self.transforms:
                img, label = self.transforms(img, label)

            if self.start_index_with_1:              
                label = label - 1
                label[label == -1] = self.ignore_index
            return img, label.astype('int64')
        else:
            img = cv2.imread(self.data[idx]).astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transforms:
                img = self.transforms(img)  
            return img

    def __len__(self):
        return len(self.data)

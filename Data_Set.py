import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2

class MyDataSet(Dataset):

    def __init__(self,  path, Cuda='cuda:0'):
        super(MyDataSet,self).__init__()
        self.base_path = path
        self.device = torch.device(Cuda if torch.cuda.is_available() else 'cpu')
        with open(path+'/SegNet/CamVid/train.txt') as f:
            txt = f.read()
            txt = str.split(txt,'\n')
            self.path = [str.split(line,' ') for line in txt]
            self.path = self.path[0:2]
        self.image_H = 360
        self.image_W = 480
        self.label_num = 12
        self.image_path = [data[0] for data in self.path]
        self.label_path = [data[1] for data in self.path]
        for index in range(len(self.image_path)):
            self.image_path[index] = self.base_path + self.image_path[index]
            #[15:]这样的话，是将最后15号以后所有的都算上
            #[15:-1]等价于[15,30],录入的是左开右闭的字符串，最后一个字是不算的
            self.label_path[index] = self.base_path + self.label_path[index]
        print(self.label_path)

    def __getitem__(self, index):
        image = self.read_image(index)
        label = self.read_label(index)
        image, label = image.to(self.device), label.to(self.device)
        return image, label

    def __len__(self):
        return len(self.path)

    def read_image(self, index):
        image = cv2.imread(self.image_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, [2, 0, 1])
        image = torch.tensor(image,dtype=torch.float)
        return image

    def read_label(self, index):
        label = cv2.imread(self.label_path[index])[:,:,0]
        label = torch.tensor(label,dtype=torch.long)
        return label
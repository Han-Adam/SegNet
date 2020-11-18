import os
import numpy as np
import cv2

import torch

PATH = os.getcwd()
print(PATH)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtCOlor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image,[2,0,1])
    image = torch.from_numpy(np.array(image,dtype='float32'))
    image = torch.unsqueeze(image,0)
    image.to(DEVICE)
    return image

segnet = torch.load(PATH+'/Model/model1.pth')
image = imread(PATH+'/SegNet/CamVid/test/0001TP_008550.png')
output = segnet(image)
output = torch.squeeze(output,0)
output = output.detach.numpy()
output = np.argmax(output,axis=0)
np.save(PATH+'/Test_image/1.npy')
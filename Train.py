from Data_Set import MyDataSet
from My_Module import SegNet
# third-party library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os

PATH = os.getcwd()+'/'

def imread(Device, path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image,[2,0,1])
    image = torch.from_numpy(np.array(image,dtype='float32'))
    image = torch.unsqueeze(image,0)
    image.to(Device)
    return image

def Train(Path, Model_name, Cuda='cuda:0', lr = 0.01,Batch_size = 1,Epoch = 1, Num_class = 12):
    segnet = SegNet(num_classes=Num_class)
    device = torch.device(Cuda if torch.cuda.is_available() else 'cpu')
    segnet.to(device)
    train_data = MyDataSet(path=Path, Cuda=Cuda)

    optimizer = torch.optim.Adam(segnet.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)

    for epoch in range(Epoch):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            output = segnet(b_x)  # cnn output
            print(output.shape,b_y.shape)
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if (step % 10) == 0:
                print("epoch=" + str(epoch) + ", step=" + str(step))

    torch.save(segnet,Path+'/Model/'+Model_name+'.pth')

def Predict(Path, Cuda='cuda:0'):
    segnet = torch.load(Path+'/Model/model1.pth')
    Device = torch.device(Cuda if torch.cuda.is_available() else 'cpu')
    segnet.to(Device)

    with open(Path + '/SegNet/CamVid/val.txt') as f:
        txt = f.read()
        txt = str.split(txt, '\n')
        path = [str.split(line, ' ') for line in txt]
        path = [data[0] for data in path]
        name = [(str.split(data,'/'))[-1] for data in path]
        name = [(str.split(data,'.'))[0] for data in name]

    for i in range(len(path)):
        image = imread(Device=Device, path= Path+ path[i])
        output = segnet(image)
        output = torch.squeeze(output, 0)
        output = output.detach().numpy()
        output = np.argmax(output, axis=0)
        np.save(Path + '/Test_image/'+name[i]+'.npy', output)

Train(Path = PATH, Model_name= 'model1', Cuda='cpu')
#Predict(Path = PATH, Cuda= 'cpu')

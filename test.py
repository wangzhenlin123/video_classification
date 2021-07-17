# python imports
import os,time,glob
# external imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
# model imports
from model import ResNet34_3D
from dataset import VideoDataset




def test():
    load_iters = 0   # 加载之前训练的模型(指定迭代次数)

    model = ResNet34_3D(9)    # 实例化模型

    # 加载权重
    if(load_iters!=0):
        pre=torch.load(os.path.join('./Checkpoint',str(load_iters)+'.pth'))
        model.load_state_dict(pre)

    model.eval()
    model = model.cuda()

    loss_fn = nn.BCELoss()

    # 加载训练数据
    dataset = VideoDataset(train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    loss_mean = 0.0
    iters = 0.0
    for img,label in dataloader:
        iters += 1
        # 数据存入gpu
        img = img.cuda()
        label = label.cuda()
        # 模型预测
        output = model(img)
        # 计算loss
        loss = loss_fn(output, label)
        loss_mean += loss.item()
    loss = loss_mean/iters
    print('测试集loss：',loss)


if __name__ == "__main__":
    test()
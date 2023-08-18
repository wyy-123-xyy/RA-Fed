#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
# 非 gnn 使用下面import
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
# from sklearn import metrics
# from sklearn.neighbors import kneighbors_graph

# # gnn 使用如下import
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader


'''class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label'''


class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        #self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                #print(images.size())
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.zero_grad()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class ag_LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        #self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=False)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images[0].to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                #print(images.size())
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.zero_grad()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class gcn_LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        #self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
        # 将灰度图转化为graph
        data_list=[]
        for images, lables in dataset:
            images = images.reshape([28,28])
            # print(images.shape)
            # print(lables)
            data = np.where(images < 0.4, -1, 1000)
            img = np.pad(data, [(2, 2), (2, 2)], "constant", constant_values=(-1))
            cnt = 0
            # 一整张图像中，图node的像素点从0到N进行标号，其实就是表示N个node。
            for i in range(2, 30):
                for j in range(2, 30):
                    if img[i][j] != -1:
                        img[i][j] = cnt
                        cnt+=1
            
            edges = []
            features = np.zeros((cnt, 2))
            # 然后就需要边信息，边具体如何找呢，就是对所有node进行循环，然后去找每个node周围的8个像素点，
            for i in range(2, 30):
                for j in range(2, 30):
                    if img[i][j] == -1:
                        continue
                    #8近傍に該当する部分を抜き取る。
                    filter = img[i-2:i+3, j-2:j+3].flatten() # 返回一个一维数组
                    filter1 = filter[[6, 7, 8, 11, 13, 16, 17, 18]] # i,j周围的八个像素点
                    features[filter[12]][0] = i-2  # filter[12]是该像素点
                    features[filter[12]][1] = j-2

                    for neighbor in filter1:
                        if not neighbor == -1:
                            edges.append([filter[12], neighbor])
            edge=torch.tensor(np.array(edges).T,dtype=torch.long)
            x=torch.tensor(np.array(features)/28,dtype=torch.float)

            d=Data(x=x,edge_index=edge.contiguous(),t=int(lables))
            data_list.append(d)

        self.ldr_train = DataLoader(data_list, batch_size=self.args.local_bs, shuffle=False)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                batch=batch.to(self.args.device)           
                net.zero_grad()
                #print(images.size())
                log_probs = net(batch)
                loss = self.loss_func(log_probs, batch.t)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(batch), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.zero_grad()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
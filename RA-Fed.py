#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import data_utils
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, get_user_typep, get_local_wmasks, get_local_bmasks
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg2,FedAvg3
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from  fvcore.nn import flop_count_table
from fvcore.nn import parameter_count
import torch
import torch.nn as nn
from torchstat import stat
seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

def get_sub_paras(w_glob, wmask, bmask):
    w_l = copy.deepcopy(w_glob)
    w_l['layer_input.weight'] = w_l['layer_input.weight'] * wmask
    # w_l['layer_input.bias'] = w_l['layer_input.bias'] * bmask

    return w_l

"""
AvgAll:

leave bias alone,
use FedAvg2

"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.epochs = 300
    alpha = args.alpha
    model_training_typep=args.model_training_typep

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("===============IID=DATA=======================")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("===========NON==ID=DATA=======================")
            ###所有client的数据   dict_users[i]
            #dict_users = mnist_noniid(dataset_train, args.num_users)
            # args.epochs = 500
            # print(args.epochs)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
            # args.epochs = 500
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        '''net_2 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_3 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_4 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        net_51 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_52 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_53 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)'''

    else:
        exit('Error: unrecognized model')
    print(net_glob)
    '''inp = torch.randn(784,1).to(args.device)
    flops1 = FlopCountAnalysis(net_glob, inp)
    print(flops1.total())
    print(flop_count_table(flops1))
    stat(net_glob, (1,28,28))   # 统计模型的参数量和FLOPs，（3,244,244）是输入图像的size'''
    net_glob.train()
    net_2=copy.deepcopy(net_glob)
    net_2.train()
    net_3=copy.deepcopy(net_glob)
    net_3.train()
    net_4=copy.deepcopy(net_glob)
    net_4.train()
    net_5=copy.deepcopy(net_glob)
    net_5.train()
    net_6=copy.deepcopy(net_glob)
    net_6.train()
    net_7=copy.deepcopy(net_glob)
    net_7.train()
    net_8=copy.deepcopy(net_glob)
    net_8.train()
    net_9=copy.deepcopy(net_glob)
    net_9.train()
    net_10=copy.deepcopy(net_glob)
    net_10.train()
    net_11=copy.deepcopy(net_glob)
    net_11.train()
    net_12=copy.deepcopy(net_glob)
    net_12.train()
    net_13=copy.deepcopy(net_glob)
    net_13.train()
    net_14=copy.deepcopy(net_glob)
    net_14.train()
    net_15=copy.deepcopy(net_glob)
    net_15.train()
    print("***********MODIFIED: PRUNING WEIGHTS ONLY******************")
    # copy weights

    # Ranking the paras
    # w_glob['layer_input.weight'].view(-1).view(200, 784)  #156800 = 39200 * 4
    # Smallest  [0,39200], [39200,78400], [78400,117600],[117600, 156800] (largest)
    """
1 w_net      X              X                X               X
2 net_2      X              X                                X
3 net_3      X                               X               X
4 net_4                     X                X               X

5 net_51     X                                               X
6 net_52                    X                                X
7 net_53                                     X               X
    """
    #    [0,50]         [50,100]         [100,150]        [150,200]
    w_glob = net_glob.state_dict()
    print("-------------------------------")
    print(w_glob.keys())
    starting_weights = copy.deepcopy(w_glob)
    # ABS OR NO ABS ?
    #wranks = torch.argsort(w_glob['layer_input.weight'].view(-1))
    '''wranks = torch.zeros(w_glob['layer_input.weight'].view(-1).size(),dtype=torch.int).to(args.device)
    for i in range(0,156800):
        wranks[i]=i'''
    wranks = torch.argsort(torch.absolute(w_glob['layer_input.weight'].view(-1)))
    local_w_masks,opp_local_w_masks = get_local_wmasks(wranks)


    # branks = torch.argsort(w_glob['layer_input.bias'])
    branks = torch.argsort(torch.absolute(w_glob['layer_input.bias']))
    local_b_masks = get_local_bmasks(branks)

    w_n2 = get_sub_paras(w_glob, local_w_masks[1], local_b_masks[1])
    net_2.load_state_dict(w_n2)
    w_n3 = get_sub_paras(w_glob, local_w_masks[2], local_b_masks[2])
    net_3.load_state_dict(w_n3)
    w_n4 = get_sub_paras(w_glob, local_w_masks[3], local_b_masks[3])
    net_4.load_state_dict(w_n4)
    w_n5 = get_sub_paras(w_glob, local_w_masks[4], local_b_masks[4])
    net_5.load_state_dict(w_n5)
    w_n6 = get_sub_paras(w_glob, local_w_masks[5], local_b_masks[5])
    net_6.load_state_dict(w_n6)
    w_n7 = get_sub_paras(w_glob, local_w_masks[6], local_b_masks[6])
    net_7.load_state_dict(w_n7)
    w_n8 = get_sub_paras(w_glob, local_w_masks[7], local_b_masks[7])
    net_8.load_state_dict(w_n8)
    w_n9 = get_sub_paras(w_glob, local_w_masks[8], local_b_masks[8])
    net_9.load_state_dict(w_n9)
    w_n10 = get_sub_paras(w_glob, local_w_masks[9], local_b_masks[9])
    net_10.load_state_dict(w_n10)
    w_n11 = get_sub_paras(w_glob, local_w_masks[10], local_b_masks[10])
    net_11.load_state_dict(w_n11)
    w_n12 = get_sub_paras(w_glob, local_w_masks[11], local_b_masks[11])
    net_12.load_state_dict(w_n12)
    w_n13 = get_sub_paras(w_glob, local_w_masks[12], local_b_masks[12])
    net_13.load_state_dict(w_n13)
    w_n14 = get_sub_paras(w_glob, local_w_masks[13], local_b_masks[13])
    net_14.load_state_dict(w_n14)
    w_n15 = get_sub_paras(w_glob, local_w_masks[14], local_b_masks[14])
    net_15.load_state_dict(w_n15)
    setting_50=[6,7,8,9,10,11]
    setting_25=[12,13,14,15]
    setting_arrays = [
        # [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],   #0
        [15, 15, 12, 12, 13, 13, 12, 12, 14, 14]             # 2
        # [14, 14, 14, 14, 12, 13, 15, 15, 15, 15],    #1
        # [14, 14, 14, 14, 14, 15, 15, 15, 15, 15],    #5
    ]
    train_data=[]
    for i in range(0,args.num_users):
        client_data = data_utils.read_client_data('mnist_'+str(alpha), i, is_train=True)
        # print(client_data)
        train_data.append(client_data)
    setting_array = setting_arrays[0]
    '''for setting_array in setting_arrays:
        net_glob.load_state_dict(starting_weights)'''
    avg_test_acc=[]
    avg_train_loss=[]
    avg_test_loss=[]
    for i in range(0,1):
        net_glob.load_state_dict(starting_weights)
        w_glob=copy.deepcopy(starting_weights)
        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        # if args.all_clients:
        #     print("Aggregation over all clients")
        #     w_locals = [w_glob for i in range(args.num_users)]
        setting=args.method+str(args.epochs)+"_"+str(alpha)+"_"+str(args.lr)
        pic_name = './result/{}_{}_{}_{}_lep{}_iid{}_{}.pdf'.format(args.dataset, args.model_training_typep, args.model,
                                                                    args.epochs, args.local_ep, args.iid, setting)
        txt_name = './result/{}_{}_{}_{}_lep{}_iid{}_{}.txt'.format(args.dataset, args.model_training_typep, args.model,
                                                                    args.epochs, args.local_ep, args.iid, setting)
        npy_name = './result/{}_{}_{}_{}_lep{}_iid{}_{}.npy'.format(args.dataset, args.model_training_typep, args.model,
                                                                    args.epochs, args.local_ep, args.iid, setting)
##########################TEMP USE FOR STRATING LOSS
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        iter = -1
        '''with open(txt_name, 'a+') as f:
            print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
        print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
        print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))'''
        net_glob.train()
        ##########################TEMP USE FOR STRATING LOSS
        setting_array_tmp=setting_array
        #C=[torch.zeros(self_grad.size(), dtype=self_grad[0].dtype).to(self.device) for i in range(0,5)]
        '''mlp_keys=['layer_input.weight', 'layer_input.bias', 'layer_hidden.weight', 'layer_hidden.bias']
        latest_client_update=[]
        start_client_update=[]
        for i in range(0,4):
            start_client_update.append(torch.zeros(starting_weights[mlp_keys[0]].size(), dtype=starting_weights[mlp_keys[0]][0].dtype))
        for i in range(args.num_users):
            latest_client_update.append(start_client_update)'''
        epoch_train_loss=[]
        epoch_test_loss=[]
        epoch_train_acc=[]
        epoch_test_acc=[]
        for iter in range(args.epochs):
            print("epoch "+str(iter+1))
            '''if iter==0:
                setting_array=[1, 1, 1, 1, 1, 1, 5, 5, 5, 2]
            else:
                setting_array=setting_array_tmp'''

            if iter > 0:  # >=5 , %5, % 50, ==5
                w_glob = net_glob.state_dict()

                # ABS OR NO ABS
                # wranks = torch.argsort(torch.absolute(w_glob['layer_input.weight'].view(-1)))
                wranks = torch.argsort(w_glob['layer_input.weight'].view(-1))
                '''wranks = torch.zeros(w_glob['layer_input.weight'].view(-1).size(),dtype=torch.int).to(args.device)
                for i in range(0,156800):
                    wranks[i]=i'''
                local_w_masks,opp_local_w_masks = get_local_wmasks(wranks)
                branks = torch.argsort(torch.absolute(w_glob['layer_input.bias']))
                local_b_masks = get_local_bmasks(branks)
                '''inp = torch.randn(784,1).to(args.device)
                flops1 = FlopCountAnalysis(net_glob, inp)
                print(flops1.total())
                print(flop_count_table(flops1))
                stat(net_glob, (1,28,28))   # 统计模型的参数量和FLOPs，（3,244,244）是输入图像的size'''
                w_n2 = get_sub_paras(w_glob, local_w_masks[1], local_b_masks[1])
                net_2.load_state_dict(w_n2)
                #stat(net_2, (1,28,28)) 
                '''flops2 = FlopCountAnalysis(net_2, inp)
                print(flops2.total())
                print(flop_count_table(flops2))'''
                w_n3 = get_sub_paras(w_glob, local_w_masks[2], local_b_masks[2])
                net_3.load_state_dict(w_n3)
                '''flops3 = FlopCountAnalysis(net_3, inp)
                flops3.total()
                print(flop_count_table(flops3))'''
                #stat(net_3, (1,28,28)) 
                w_n4 = get_sub_paras(w_glob, local_w_masks[3], local_b_masks[3])
                net_4.load_state_dict(w_n4)
                w_n5 = get_sub_paras(w_glob, local_w_masks[4], local_b_masks[4])
                net_5.load_state_dict(w_n5)
                w_n6 = get_sub_paras(w_glob, local_w_masks[5], local_b_masks[5])
                net_6.load_state_dict(w_n6)
                w_n7 = get_sub_paras(w_glob, local_w_masks[6], local_b_masks[6])
                net_7.load_state_dict(w_n7)
                w_n8 = get_sub_paras(w_glob, local_w_masks[7], local_b_masks[7])
                net_8.load_state_dict(w_n8)
                w_n9 = get_sub_paras(w_glob, local_w_masks[8], local_b_masks[8])
                net_9.load_state_dict(w_n9)
                w_n10 = get_sub_paras(w_glob, local_w_masks[9], local_b_masks[9])
                net_10.load_state_dict(w_n10)
                w_n11 = get_sub_paras(w_glob, local_w_masks[10], local_b_masks[10])
                net_11.load_state_dict(w_n11)
                w_n12 = get_sub_paras(w_glob, local_w_masks[11], local_b_masks[11])
                net_12.load_state_dict(w_n12)
                w_n13 = get_sub_paras(w_glob, local_w_masks[12], local_b_masks[12])
                net_13.load_state_dict(w_n13)
                w_n14 = get_sub_paras(w_glob, local_w_masks[13], local_b_masks[13])
                net_14.load_state_dict(w_n14)
                w_n15 = get_sub_paras(w_glob, local_w_masks[14], local_b_masks[14])
                net_15.load_state_dict(w_n15)
                """
                net_glob : full net
                net_2~net_4: 75% net 
                net_51 ~ net_53 50% net     
                """
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            '''m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)'''
            idxs_users=[i for i in range(0,args.num_users)]
            print(idxs_users)

            type_array = []
            count50=0
            count25=0
            for id, idx in enumerate(idxs_users):
                # typep=1

                if model_training_typep=="L":
                    if iter==0:   #第一次使用全局模型进行训练
                        typep = 1
                    elif iter%10==0:  #确保non-fullcover
                        typep=11  #50
                    else:
                        typep=np.random.choice(setting_50)
                elif model_training_typep=="S":
                    if iter == 0:  # 第一次使用全局模型进行训练
                        typep = 1
                    elif iter % 10 == 0:  # 确保non-fullcover
                        typep=15  #25%
                    else:
                        typep=np.random.choice(setting_25)
                elif model_training_typep=="MIX":
                    if iter == 0:  # 第一次使用全局模型进行训练
                        typep = 1
                    elif iter % 10 == 0:  # 确保non-fullcover
                        if count50<5 and count25<5:
                            opt=np.random.choice([50,25])
                            if opt==50:
                                typep=11
                                count50=count50+1
                            else:
                                typep=15
                                count25=count25+1
                        elif count50<5:
                            typep=11
                            count50=count50+1
                        else:
                            typep=15
                            count25=count25+1
                    else:
                        if count50<5 and count25<5:
                            opt=np.random.choice([50,25])
                            if opt==50:
                                typep=np.random.choice(setting_50)
                                count50=count50+1
                            else:
                                typep=np.random.choice(setting_25)
                                count25=count25+1
                        elif count50<5:
                            typep=np.random.choice(setting_50)
                            count50=count50+1
                        else:
                            typep=np.random.choice(setting_25)
                            count25=count25+1
                elif model_training_typep=="FULL":
                    typep = 1
                else:
                    print("error")
                    sys.exit(0)


                local = LocalUpdate(args=args, dataset=train_data[idx])
                #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                new_update=[]
                if typep == 1:
                    type_array.append(1)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_glob[mlp_keys[i]])'''
                elif typep == 2:
                    type_array.append(2)
                    w, loss = local.train(net=copy.deepcopy(net_2).to(args.device))
                    w = get_sub_paras(w, local_w_masks[1], local_b_masks[1])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n2[mlp_keys[i]])'''
                elif typep == 3:
                    type_array.append(3)
                    w, loss = local.train(net=copy.deepcopy(net_3).to(args.device))
                    w = get_sub_paras(w, local_w_masks[2], local_b_masks[2])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n3[mlp_keys[i]])'''
                elif typep == 4:
                    type_array.append(4)
                    w, loss = local.train(net=copy.deepcopy(net_4).to(args.device))
                    w = get_sub_paras(w, local_w_masks[3], local_b_masks[3])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n4[mlp_keys[i]])'''
                elif typep == 5:
                    type_array.append(5)
                    w, loss = local.train(net=copy.deepcopy(net_5).to(args.device))
                    w = get_sub_paras(w, local_w_masks[4], local_b_masks[4])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n5[mlp_keys[i]])'''
                elif typep == 6:
                    type_array.append(6)
                    w, loss = local.train(net=copy.deepcopy(net_6).to(args.device))
                    w = get_sub_paras(w, local_w_masks[5], local_b_masks[5])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n6[mlp_keys[i]])'''
                elif typep == 7:
                    type_array.append(7)
                    w, loss = local.train(net=copy.deepcopy(net_7).to(args.device))
                    w = get_sub_paras(w, local_w_masks[6], local_b_masks[6])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n7[mlp_keys[i]])'''
                elif typep == 8:
                    type_array.append(8)
                    w, loss = local.train(net=copy.deepcopy(net_8).to(args.device))
                    w = get_sub_paras(w, local_w_masks[7], local_b_masks[7])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n8[mlp_keys[i]])'''
                elif typep == 9:
                    type_array.append(9)
                    w, loss = local.train(net=copy.deepcopy(net_9).to(args.device))
                    w = get_sub_paras(w, local_w_masks[8], local_b_masks[8])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n9[mlp_keys[i]])'''
                elif typep == 10:
                    type_array.append(10)
                    w, loss = local.train(net=copy.deepcopy(net_10).to(args.device))
                    w = get_sub_paras(w, local_w_masks[9], local_b_masks[9])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n9[mlp_keys[i]])'''
                elif typep == 11:
                    type_array.append(11)
                    w, loss = local.train(net=copy.deepcopy(net_11).to(args.device))
                    w = get_sub_paras(w, local_w_masks[10], local_b_masks[10])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n11[mlp_keys[i]])'''
                elif typep == 12:
                    type_array.append(12)
                    w, loss = local.train(net=copy.deepcopy(net_12).to(args.device))
                    w = get_sub_paras(w, local_w_masks[11], local_b_masks[11])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n11[mlp_keys[i]])'''
                elif typep == 13:
                    type_array.append(13)
                    w, loss = local.train(net=copy.deepcopy(net_13).to(args.device))
                    w = get_sub_paras(w, local_w_masks[12], local_b_masks[12])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n11[mlp_keys[i]])'''
                elif typep == 14:
                    type_array.append(14)
                    w, loss = local.train(net=copy.deepcopy(net_14).to(args.device))
                    w = get_sub_paras(w, local_w_masks[13], local_b_masks[13])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n11[mlp_keys[i]])'''
                elif typep == 15:
                    type_array.append(15)
                    w, loss = local.train(net=copy.deepcopy(net_15).to(args.device))
                    w = get_sub_paras(w, local_w_masks[14], local_b_masks[14])
                    #####更新值
                    '''for i in range(4):
                        new_update.append(w[mlp_keys[i]]-w_n11[mlp_keys[i]])'''
                # if args.all_clients:
                #     # w_locals[idx] = copy.deepcopy(w)
                #     pass
                # else:
                ##################更新最新的全局更新
                ###最新的部分更新new_update
                ##typep
                ##保留latest_client_update[idx][0]  本轮mask0 的部分 
                # #以上值加上new_update  [idx][0]..[3] 
                '''if typep>1:
                    latest_client_update[idx][0]=latest_client_update[idx][0].to(args.device)*opp_local_w_masks[typep-1]+new_update[0]
                else:
                    latest_client_update[idx][0]=new_update[0]
                for i in range(1,4):
                    latest_client_update[idx][i]=new_update[i]
                client_w=[]
                for i in range(0,4):
                    client_w.append(copy.deepcopy(w_glob[mlp_keys[i]]+latest_client_update[idx][i]))'''

                w_locals.append(copy.deepcopy(w))
                ###############loss也得改！！！！！！！！！！！！！！！！！！！！
                loss_locals.append(copy.deepcopy(loss))

            # with open(txt_name, 'a+') as f:
            #     print(type_array, file=f)
            print(type_array)
            # FOR ITER

            # update global weights
            w_glob = FedAvg2(w_glob,w_locals, type_array, local_w_masks, local_b_masks)
            #w_glob = FedAvg2(w_glob,w_locals, type_array, local_w_masks, local_b_masks)
            #w_glob = FedAvg(w_locals, type_array, local_w_masks, local_b_masks)
            '''for i in range (0,4):
                w_glob[mlp_keys[i]]=w_g[i]'''
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # with open(txt_name, 'a+') as f:
            #     print(loss_locals, file=f)
            print(loss_locals)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            epoch_train_loss.append(loss_avg)
            '''with open(txt_name, 'a+') as f:
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg), file=f)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))'''
            loss_train.append(loss_avg)

            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            '''with open(txt_name, 'a+') as f:
                print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
                print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
            print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))'''
            epoch_test_loss.append(loss_test)
            epoch_test_acc.append(acc_test.item())
            print(acc_test.item())
            net_glob.train()
        avg_train_loss.append(epoch_train_loss)
        avg_test_loss.append(epoch_test_loss)
        avg_test_acc.append(epoch_test_acc)
    # plot loss curve
    avg_epoch_train_loss=[]
    avg_epoch_test_loss=[]
    avg_epoch_test_acc=[]
    max_acc=0
    for i in range(args.epochs):
        sum=0
        for j in range(0,1):
            sum=sum+avg_train_loss[j][i]
        sum=sum/1
        avg_epoch_train_loss.append(sum)
    for i in range(args.epochs):
        sum=0
        for j in range(0,1):
            sum=sum+avg_test_loss[j][i]
        sum=sum/1
        avg_epoch_test_loss.append(sum)
    for i in range(args.epochs):
        sum=0
        for j in range(0,1):
            if max_acc<avg_test_acc[j][i]:
                max_acc=avg_test_acc[j][i]
            sum=sum+avg_test_acc[j][i]
        sum=sum/1
        avg_epoch_test_acc.append(sum)
    with open(txt_name, 'w') as f:
        for i in range(args.epochs):
            print(avg_epoch_train_loss[i],avg_epoch_test_loss[i],avg_epoch_test_acc[i], file=f)
        print(max_acc,file=f)
    #print('{3f} {3f} {3f}'.format(avg_epoch_train_loss[i],avg_epoch_test_loss[i],avg_epoch_test_acc[i]), file=f)

    plt.figure()
    plt.plot(range(len(avg_epoch_train_loss)), avg_epoch_train_loss)
    plt.ylabel('train_loss')
    plt.savefig(pic_name)
    np.save(npy_name, avg_epoch_train_loss)

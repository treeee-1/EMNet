'''
# MIT License
#
# Copyright (c) 2021 Bubbliiiing
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.gate_ml import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils_fit import mutifit_one_epoch


if __name__ == "__main__":

    Cuda            = True
    num_classes     = 8
    backbone        = "mobilenet"
    pretrained      = False
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    downsample_factor   = 16
    input_shape         = [512, 512]

    Init_Epoch          = 0
    Freeze_Epoch        =1
    Freeze_batch_size   = 5
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 6

    Freeze_Train        = True

    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4

    lr_decay_type       = 'cos'

    save_period         = 1

    save_dir            = 'logs'
    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = False
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)

    num_workers         = 4
    ngpus_per_node  = torch.cuda.device_count()

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    scaler = None
    model_train     = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4

        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, False, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4

                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            mutifit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
            cls_weights, num_classes, save_period, save_dir)
        loss_history.writer.close()

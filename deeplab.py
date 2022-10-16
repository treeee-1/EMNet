import colorsys
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.gate_ml import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image

class DeeplabV3(object):
    _defaults = {

        "model_path"        : 'logs/6gate/ep078-loss0.089-val_loss2.419.pth',

        "num_classes"       : 8,

        "backbone"          : "mobilenet",

        "input_shape"       : [512, 512],

        "downsample_factor" : 16,

        "mix_type"          : 0,

        "cuda"              : True,
     
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            # self.colors = [ (0, 0, 0), (128, 0, 0), (255, 0, 255), (128, 128, 0), (0, 0, 255), (128, 0, 128), (255, 255, 0),
            #                 (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
            #                 (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
            #                 (128, 64, 12)]
            self.colors = [ (0, 0, 0), (128, 0, 0), (128,64, 128), (192, 0, 192), (0, 128, 0), (128, 128, 0), (64, 64, 0),
                            (64, 0, 128)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self):

        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone, downsample_factor=self.downsample_factor, pretrained=False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image,label2):

        image = cvtColor(image)
        label2 = cvtColor(label2)

        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        label2_data, nw, nh = resize_image(label2, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        label2_data = np.expand_dims(np.transpose(preprocess_input(np.array(label2_data, np.float32)), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            labels2 = torch.from_numpy(label2_data)
            if self.cuda:
                images = images.cuda()
                labels2 = labels2.cuda()

            pr = self.net(images,labels2)[0][0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            #   将新图与原图及进行混合
            # image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #   将新图片转换成Image的形式
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #   将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_miou_png(self, image,label2):

        image       = cvtColor(image)
        label2       = cvtColor(label2)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        label2_data, nw, nh  = resize_image(label2, (self.input_shape[1],self.input_shape[0]))

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        label2_data  = np.expand_dims(np.transpose(preprocess_input(np.array(label2_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            labels2 = torch.from_numpy(label2_data)

            if self.cuda:
                images = images.cuda()
                labels2 = labels2.cuda()
            pr = self.net(images,labels2)[0][0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

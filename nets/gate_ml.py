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

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenetv2 import mobilenetv2
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import cv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):

        # 32-16 256
        low_level_features1 = self.features[:2](x)
        # 16-24 128
        low_level_features2 = self.features[2:3](low_level_features1)  # 第三层 16-24
        # 24 - 32 64 64
        low_level_features3 = self.features[3:5](low_level_features2)
        # 32-64 32 32
        low_level_features4 = self.features[5:8](low_level_features3)
        # 64-96  32 32
        low_level_features5 = self.features[8:12](low_level_features4)
        # 160 32 32
        low_level_features6 = self.features[12:15](low_level_features5)
        # 320 32 32
        low_level_features7 = self.features[15:](low_level_features6)

        return low_level_features1, low_level_features2, low_level_features3, low_level_features4, \
               low_level_features5, low_level_features6, low_level_features7

'''BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
Copyright (C) 2019 NVIDIA Corporation. Towaki Takikawa, David Acuna, Varun Jampani, Sanja Fidler
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Permission to use, copy, modify, and distribute this software and its documentation
for any non-commercial purpose is hereby granted without fee, provided that the above
copyright notice appear in all copies and that both that copyright notice and this
permission notice appear in supporting documentation, and that the name of the author
not be used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
~
GatedSpatialConv2d were proposed in:
  Towaki Takikawa, David Acuna, Varun Jampani, Sanja Fidler
  Gated-SCNN: Gated Shape CNNs for Semantic Segmentation. Presented at Proceedings of the IEEE/CVF international conference on computer vision, 2019. 5229-38. 10.1109/ICCV.2019.00533. 
If you use this code, please cite:
@article{takikawa2019gated,
  title={Gated-SCNN: Gated Shape CNNs for Semantic Segmentation},
  author={Takikawa, Towaki and Acuna, David and Jampani, Varun and Fidler, Sanja},
  journal={ICCV},
  year={2019}
}
'''
class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1, momentum=0.1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1, momentum=0.1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):  # 320 - 256
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5 + 256, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, edge):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)

        conv3x3_1 = self.branch2(x)

        conv3x3_2 =  self.branch3(x)
        
        conv3x3_3 =  self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        edge_features = F.interpolate(edge, (row, col), mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((global_feature, edge_features), 1)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, out], dim=1)
        result = self.conv_cat(feature_cat)

        return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet.'.format(backbone))

        self.m1f16_160 = nn.Conv2d(16, 32, 1)

        self.n2 = nn.Conv2d(24, 1, 1)
        self.n3 = nn.Conv2d(32, 1, 1)
        self.n4 = nn.Conv2d(64, 1, 1)
        self.n5 = nn.Conv2d(96, 1, 1)
        self.n6 = nn.Conv2d(160, 1, 1)
        self.n7 = nn.Conv2d(320, 1, 1)

        self.res1 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d1 = nn.Conv2d(32, 26, 1)

        self.res2 = BasicBlock(26, 26, stride=1, downsample=None)
        self.d2 = nn.Conv2d(26, 20, 1)

        self.res3 = BasicBlock(20, 20, stride=1, downsample=None)
        self.d3 = nn.Conv2d(20, 14, 1)

        self.res4 = BasicBlock(14, 14, stride=1, downsample=None)
        self.d4 = nn.Conv2d(14, 8, 1)

        self.res5 = BasicBlock(8, 8, stride=1, downsample=None)
        self.d5 = nn.Conv2d(8, 8, 1)

        self.res6 = BasicBlock(8, 8, stride=1, downsample=None)
        self.d6 = nn.Conv2d(8, 4, 1)

        self.g1 = GatedSpatialConv2d(26, 26)
        self.g2 = GatedSpatialConv2d(20, 20)
        self.g3 = GatedSpatialConv2d(14, 14)
        self.g4 = GatedSpatialConv2d(8, 8)
        self.g5 = GatedSpatialConv2d(8, 8)
        self.g6 = GatedSpatialConv2d(4, 4)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x, y):
        H, W = x.size(2), x.size(3)
        y_size = y.size()

        low_level_features1, low_level_features2, low_level_features3, low_level_features4, \
        low_level_features5, low_level_features6, low_level_features7 = self.backbone(x)

        s2 = F.interpolate(self.n2(low_level_features2), size=(H, W), mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.n3(low_level_features3), size=(H, W), mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.n4(low_level_features4), size=(H, W), mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.n5(low_level_features5), size=(H, W), mode='bilinear', align_corners=True)
        s6 = F.interpolate(self.n6(low_level_features6), size=(H, W), mode='bilinear', align_corners=True)
        s7 = F.interpolate(self.n7(low_level_features7), size=(H, W), mode='bilinear', align_corners=True)

        m1f = F.interpolate(low_level_features1, size=(H, W), mode='bilinear', align_corners=True)
        m1f = self.m1f16_160(m1f)

        cs1 = self.res1(m1f)
        cs1 = self.d1(cs1)
        cs1 = self.g1(cs1, s2)

        cs2 = self.res2(cs1)
        cs2 = self.d2(cs2)
        cs2 = self.g2(cs2, s3)


        cs3 = self.res3(cs2)
        cs3 = self.d3(cs3)
        cs3 = self.g3(cs3, s4)

        cs4 = self.res4(cs3)
        cs4 = self.d4(cs4)
        cs4 = self.g4(cs4, s5)

        cs5 = self.res5(cs4)
        cs5 = self.d5(cs5)
        cs5 = self.g5(cs5, s6)

        cs6 = self.res6(cs5)
        cs6 = self.d6(cs6)
        cs6 = self.g6(cs6, s7)

        cs = self.fuse(cs6)

        edge_out = torch.sigmoid(cs)

        im_arr = ((y.cpu().numpy()) * 255).transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((y_size[0], 1, y_size[2], y_size[3]))
        for i in range(y_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = torch.sigmoid(acts)


        x = self.aspp(low_level_features7, acts)

        low_level_featuresl6 = self.a(low_level_features6)
        e1 = x + low_level_featuresl6
        e1 = F.interpolate(e1, size=(64, 64), mode='bilinear', align_corners=True)
        low_level_featuresl3 = self.b(low_level_features3)
        e2 = e1 + low_level_featuresl3
        e2 = F.interpolate(e2, size=(128, 128), mode='bilinear', align_corners=True)
        low_level_featuresl2 = self.c(low_level_features2)
        e3 = e2 + low_level_featuresl2

        low_level_features = self.shortcut_conv(low_level_features2)  # 16+24

        e3 = self.cat_conv(torch.cat((e3, low_level_features), dim=1))
        e3 = self.cls_conv(e3)
        e3 = F.interpolate(e3, size=(H, W), mode='bilinear', align_corners=True)
        return e3, edge_out
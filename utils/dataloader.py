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
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, random_, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.random_              = random_
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png1         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass1"), name + ".png"))
        png2         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass2"), name + ".png"))

        jpg, png1, png2    = self.get_random_data(jpg, png1,png2, self.input_shape, random = self.random_)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png1         = np.array(png1)
        png1[png1 >= self.num_classes] = self.num_classes
        
        png2         = np.array(png2)
        if png2.ndim == 3:
            png2 = np.squeeze(png2[:, :, 0])
        assert png2.ndim == 2
        png2 = png2[np.newaxis, :, :]
        png2[png2 == 0] = 0
        png2[np.logical_and(png2 > 0, png2 < 128)] = 2
        png2[png2 >= 128] = 1

        return jpg, png1,png2

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, label2,input_shape,random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        label2 = cvtColor(label2)
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))

            label2 = label2.resize((nw,nh), Image.BICUBIC)
            new_label2  = Image.new('RGB', [w, h], (128,128,128))
            new_label2.paste(label2, ((w-nw)//2, (h-nh)//2))

            return new_image, new_label,new_label2


def deeplab_dataset_collate(batch):

    images     = []
    pngs1        = []
    pngs2        = []

    for img, png1, png2 in batch:
        images.append(img)
        pngs1.append(png1)
        pngs2.append(png2)

    images     = np.array(images)
    pngs1        = np.array(pngs1)
    pngs2        = np.array(pngs2)

    return images, pngs1,pngs2

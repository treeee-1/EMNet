# EMNet
The trained model can be downloaded here: link: https://pan.baidu.com/s/12npvngSCe--y2-gCudp0Yg Extraction code: trej

## Train
1. Use the VOC format for training.
2. Place the segmentation labels and edge labels in the SegmentationClass1 and SegmentationClass2 folders under the VOC2007 folder in the VOCdevkit folder before training.
3. Place the image files in the JPEGImages folder under the VOC2007 folder in the VOCdevkit folder before training.
4. Use the utils.voc_annotation.py file to generate the corresponding txt before training.
5. Run train.py

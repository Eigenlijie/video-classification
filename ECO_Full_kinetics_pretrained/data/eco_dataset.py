#! /usr/bin/env python
# -*- coding : utf-8 -*-
"""Dataset function"""
__all__ = ["ImageNpyDataset"]


import os
import warnings
import numpy as np
from mxnet.gluon.data import dataset
from mxnet import nd


class ImageNpyDataset(dataset.Dataset):
    """A dataset for loading image numpy files stored in a folder structure like::

        root/993519950.npy
        root/973968056.npy
        root/944802660.npy
        root/874569539.npy
        root/993519950.npy
        root/995160317.npy

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::

        transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """

    def __init__(self, root, anno_file, transform=None, flag=1):
        self._root = os.path.expanduser(root)
        self._anno_file = anno_file
        self._flag = flag
        self._transform = transform
        self._exts = ['.npy']
        self._list_videos(self._root, self._anno_file)
        self.synsets = ['狗', '猫', '鼠', '兔子', '鸟', '风景', '风土人情', '穿秀', '宝宝', '男生自拍', '女生自拍', '做甜品', '做海鲜', '街边小吃',
                        '饮品', '火锅', '抓娃娃', '手势舞', '街舞', '国标舞', '钢管舞', '芭蕾舞', '广场舞', '名族舞', '绘画', '手写文字', '咖啡拉花', '沙画',
                        '史莱姆', '折纸', '编织', '发饰', '陶艺', '手机壳', '打鼓', '弹吉他', '弹钢琴', '弹古筝', '弹小提琴', '弹大提琴', '吹葫芦丝', '唱歌',
                        '游戏', '娱乐', '动画', '文字艺术配音', '瑜伽', '健身', '滑板', '篮球', '跑酷', '潜水', '台球', '足球', '羽毛球', '乒乓球', '画眉',
                        '画眼', '护肤', '唇彩', '卸妆', '美甲', '美发']

    def _list_videos(self, root, anno_file):
        self.items = []
        # anno_file = 'DatasetLabels/short_video_trainingset_annotations.txt.082902'
        anno_file_path = os.path.join(root, anno_file)
        if not os.path.exists(anno_file_path):
            warnings.warn('Annotations file %s is not exists.' % root, stacklevel=3)
        with open(anno_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            linelist = line.split(',')
            filename = linelist[0].split('.mp4')[0].strip()
            filename = filename + self._exts[0]
            if not os.path.exists(self._root + filename):
                warnings.warn("Numpy file %s is not exists." % filename)
                continue
            # just use one label
            label = int(linelist[1].strip())
            # use all labels
            # labels = []
            # for i in range(1, len(linelist)):
            #     labels.append(linelist[i].strip())

            # self.items = [('862639748.npy', 11), ('887245704.npy', 59), ('98720347.npy', 34) ... ('97957243.npy', 12)]
            self.items.append((filename, label))

    def __getitem__(self, idx):
        # print('Call function __getitem__')
        filename = self._root + self.items[idx][0]
        if not os.path.exists(filename):
            warnings.warn("Numpy file %s is not exists." % filename)
            return
        video_imgs = np.load(filename)[:16]
        # ndarray img->(B, C, W, H) eg.(16, 3, 224, 224) ------>    NDArray img->(B, W, H, C)  eg.(16, 224, 224, 3)
        video_imgs = video_imgs.transpose((0, 2, 3, 1))
        # numpy.ndarray convert to mxnet.ndarray.ndarray.NDArray
        video_imgs = nd.array(video_imgs)
        # 由于mxnet的ImageFoldDataset()方法中的__getitem__是针对单张图片的处理方法，此处不仅需要使image变成224x224x3，还需要做单张处理
        label = self.items[idx][1]
#        if self._transform is not None:
#            print('self._transform : {}'.format(self._transform))
#            trans_video_imgs = nd.zeros((224, 224, 3))
#            first_image = True
#            for i in range(video_imgs.shape[0]):
#                trans_video_img = self._transform(video_imgs[i], label)
#                trans_video_img = trans_video_img.reshape(-1, trans_video_img.shape[0], trans_video_img.shape[1], trans_video_img.shape[2])
#                if first_image:
#                    trans_video_imgs = trans_video_img
#                    first_image = False
#                else:
#                    trans_video_imgs = nd.concat(trans_video_imgs, trans_video_img)
#            return trans_video_imgs, label
        return video_imgs, label

    def __len__(self):
        return len(self.items)

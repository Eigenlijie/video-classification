#! /usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet
from mxnet import image
from mxnet import nd
from mxnet.gluon.block import Block
from mxnet.gluon.data.vision import transforms


class Compose(transforms.Compose):
    """ Sequentially composes multiple transforms. Just inherited from transforms.Compose """
    pass


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the input image with random scale and aspect ratio.

        Makes a crop of the original image with random size (default: 0.08
        to 1.0 of the original image size) and random aspect ratio (default:
        3/4 to 4/3), then resize it to the specified size.

        Parameters
        ----------
        size : int or tuple of (W, H)
            Size of the final output.
        scale : tuple of two floats
            If scale is `(min_area, max_area)`, the cropped image's area will
            range from min_area to max_area of the original image's area
        ratio : tuple of two floats
            Range of aspect ratio of the cropped image before resizing.
        interpolation : int
            Interpolation method for resizing. By default uses bilinear
            interpolation. See OpenCV's resize function for available choices.


        Inputs:
            - **data**: input tensor with (N x Hi x Wi x C) shape.

        Outputs:
            - **out**: output tensor with (N x H x W x C) shape.
        """
    def forward(self, x):
        first_image = True
        for i in range(x.shape[0]):
            # print('video_imgs[i].shape : {}'.format(x[i].shape))
            trans_video_img = image.random_size_crop(x[i], *self._args)[0]
            h, w, c = trans_video_img.shape
            trans_video_img = trans_video_img.reshape(-1, h, w, c)
            if first_image:
                trans_video_imgs = trans_video_img
                first_image = False
            else:
                trans_video_imgs = nd.concat(trans_video_imgs, trans_video_img, dim=0)

        return trans_video_imgs


class RandomFlipLeftRight(Block):
    """Randomly flip the input image left to right with a probability
    of 0.5.

    Inputs:
        - **data**: input tensor with (N x H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self):
        super(RandomFlipLeftRight, self).__init__()

    def forward(self, x):
        first_image = True
        for i in range(x.shape[0]):
            # print('video_imgs[i].shape : {}'.format(x[i].shape))
            trans_video_img = nd.image.random_flip_left_right(x[i])
            h, w, c = trans_video_img.shape
            trans_video_img = trans_video_img.reshape(-1, h, w, c)
            if first_image:
                trans_video_imgs = trans_video_img
                first_image = False
            else:
                trans_video_imgs = nd.concat(trans_video_imgs, trans_video_img, dim=0)

        return trans_video_imgs


class ToTensor(Block):
    """Converts an image NDArray to a tensor NDArray.

    Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1).

    Inputs:
        - **data**: input tensor with (N x H x W x C) shape and uint8 type.

    Outputs:
        - **out**: output tensor with (N x C x H x W) shape and float32 type.

    Examples
    --------
    >>> transformer = vision.transforms.ToTensor()
    >>> image = mx.nd.random.uniform(0, 255, (1, 4, 2, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    [[[[ 0.85490197  0.72156864]
      [ 0.09019608  0.74117649]
      [ 0.61960787  0.92941177]
      [ 0.96470588  0.1882353 ]]
     [[ 0.6156863   0.73725492]
      [ 0.46666667  0.98039216]
      [ 0.44705883  0.45490196]
      [ 0.01960784  0.8509804 ]]
     [[ 0.39607844  0.03137255]
      [ 0.72156864  0.52941179]
      [ 0.16470589  0.7647059 ]
      [ 0.05490196  0.70588237]]]]
    <NDArray 3x4x2 @cpu(0)>
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x):
        first_image = True
        for i in range(x.shape[0]):
            # print('video_imgs[i].shape : {}'.format(x[i].shape))
            trans_video_img = nd.image.to_tensor(x[i])
            h, w, c = trans_video_img.shape
            trans_video_img = trans_video_img.reshape(-1, h, w, c)
            if first_image:
                trans_video_imgs = trans_video_img
                first_image = False
            else:
                trans_video_imgs = nd.concat(trans_video_imgs, trans_video_img, dim=0)

        return trans_video_imgs


class Normalize(Block):
    """Normalize an tensor of shape (C x H x W) with mean and
    standard deviation.

    Given mean `(m1, ..., mn)` and std `(s1, ..., sn)` for `n` channels,
    this transform normalizes each channel of the input tensor with::

        output[i] = (input[i] - mi) / si

    If mean or std is scalar, the same value will be applied to all channels.

    Parameters
    ----------
    mean : float or tuple of floats
        The mean values.
    std : float or tuple of floats
        The standard deviation values.


    Inputs:
        - **data**: input tensor with (N x C x H x W) shape.

    Outputs:
        - **out**: output tensor with the shape as `data`.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self._mean = mean
        self._std = std

    def forward(self, x):
        first_image = True
        for i in range(x.shape[0]):
            trans_video_img = nd.image.normalize(x[i], self._mean, self._std)
            h, w, c = trans_video_img.shape
            trans_video_img = trans_video_img.reshape(-1, h, w, c)
            if first_image:
                trans_video_imgs = trans_video_img
                first_image = False
            else:
                trans_video_imgs = nd.concat(trans_video_imgs, trans_video_img, dim=0)

        return trans_video_imgs

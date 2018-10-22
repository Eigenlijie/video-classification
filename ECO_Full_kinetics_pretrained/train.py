#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import argparse
import logging
import warnings
import math
import random
import transforms
import metric
import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data import DataLoader
from data import eco_dataset
from network import eco_full


def get_transform(param):
    """ Transform input into required image shape"""
    if 'train' == param:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_train
    elif 'test' == param:
        transform_test = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_test


def get_schedule(epochs, train_data, batch_size):
    """ learning scheduler """
    # just refactor 3 times, refactor is 10
    decay_interval = epochs / 3
    steps_epochs = [i for i in range(1, epochs, decay_interval)]
    iterations_per_epoch = math.ceil(len(train_data) / batch_size)
    steps_iterations = [s*iterations_per_epoch for s in steps_epochs]
    print("Learning rate drops after iterations: {}".format(steps_iterations))

    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.1)
    # sgd_optimizer = mx.optimizer.SGD(learning_rate=0.001, lr_scheduler=schedule, momentum=momentum, wd=weight_decay)
    # trainer = gluon.Trainer(params=net.collect_params(), optimizer=sgd_optimizer)
    return schedule


def test(net, val_data, ctx):
    """ Evaluate the result"""
    metric_acc = metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = []
        data = data[0]
        label = label[0]
        for idx in range(data.shape[0]):
            outputs.append(net(data[i], ctx))
        metric_acc.update(label, outputs)

    return metric_acc.get()


def split_image_dataset(train_path, val_path, anno_file, val_size=0.2):
    """ Split annotation file into training and validation dataset. """
    # anno_file = 'DatasetLabels/short_video_trainingset_annotations.txt.082902'
    train_anno_file = 'train_annotations.txt'
    val_anno_file = 'val_annotation.txt'
    if os.path.exists(train_path+train_anno_file) and os.path.exists(val_path+val_anno_file):
        return train_anno_file, val_anno_file
    if not os.path.exists(anno_file):
        warnings.warn('Annotations file %s is not exists.' % anno_file, stacklevel=3)
    with open(anno_file, 'r') as f:
        lines = f.readlines()

    train_lines = lines
    val_lines_str = ''
    train_lines_str = ''
    val_num = int(len(lines)*val_size)
    for i in range(val_num):
        randomIndex = int(random.uniform(0, len(train_lines)))
        val_lines_str += train_lines[randomIndex]
        del train_lines[randomIndex]

    for i in range(len(train_lines)):
        train_lines_str += train_lines[i]

    # training set
    with open(train_path + train_anno_file, 'w') as f:
        f.write(train_lines_str)
    # validation set
    with open(val_path + val_anno_file, 'w') as f:
        f.write(val_lines_str)

    return train_anno_file, val_anno_file


def get_latest_params_file(pretrained_path):
    """ Get latest saved model parameters file. """
    filename_list = sorted(os.listdir(pretrained_path), key=lambda x: os.path.getmtime(os.path.join(pretrained_path, x)))
    if not filename_list:
        return
    filename_list = [item for item in filename_list if '.params' in item]
    return filename_list[len(filename_list) - 1]


def train_net(train_path, val_path, anno_file,
              num_class, batch_size,
              pretrained, pretrained_path, epochs,
              ctx, learning_rate, weight_decay,
              optimizer, momentum,
              lr_refactor_steps, lr_refactor_ratio,
              log_file, tensorboard,
              num_workers, per_device_batch_size):
    """ Training network """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # split training dataset into training and validation dataset
    train_anno_file, val_anno_file = split_image_dataset(train_path, val_path, anno_file)
    # load dataset
    train_data = DataLoader(
        eco_dataset.ImageNpyDataset(train_path, train_anno_file).transform_first(get_transform('train')),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = DataLoader(
        eco_dataset.ImageNpyDataset(val_path, val_anno_file).transform_first(get_transform('test')),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # build network
    net = eco_full.eco_full()

    # pre-train model
    if pretrained:
        logger.info("Start training from pretrained model {}".format(pretrained))
        params_file = get_latest_params_file(pretrained_path)
        if not params_file:
            logger.info("No params file exist, the net will be initialized by Xavier")
            net.collect_params().initialize(mx.init.Xavier(), ctx)
            net.hybridize()
        else:
#            logger.info("Initialize network by symbol parameters.")
#            net = gluon.SymbolBlock.imports("eco_gluon_to_symbol-symbol.json",
#                        ["data"], "eco_gluon_to_symbol-0000.params", ctx=mx.gpu())

            logger.info("Initialize network by %s" % params_file)
            net.load_parameters('/home/lijie/ECO_Full_kinetics_pretrained/model/'+params_file, ctx)
            net.hybridize()
    else:
        net.collect_params().initialize(mx.init.Xavier(), ctx)
        net.hybridize()

    # learning rate refactor steps
    if lr_refactor_steps is None:
        decay_interval = int(epochs / 3)
        lr_refactor_steps = [i for i in range(1, epochs, decay_interval)]
    else:
        lr_refactor_steps = [int(i.strip()) for i in lr_refactor_steps.split(',')]

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': learning_rate, 'momentum': momentum, 'wd': weight_decay})

    metric_acc = metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_counter = 0
    num_batch = len(train_data)

    for epoch in range(epochs):
        epoch_start = time.time()
        if lr_counter < len(lr_refactor_steps) and epoch == lr_refactor_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_refactor_ratio)
            lr_counter += 1
        train_loss = 0
        metric_acc.reset()

        for i, batch in enumerate(train_data):
            batch_start = time.time()
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                # print('data length : {}'.format(len(data)))
                outputs = []
                data = data[0]
                label = label[0]
                for idx in range(data.shape[0]):
                    outputs.append(net(data[idx]))
                loss = 0
                for yhat, y in zip(outputs, label):
                    loss = loss + mx.nd.mean(L(yhat, y))
                loss.backward()
            # for l in loss:
            #     l.backward()

            trainer.step(batch_size, ignore_stale_grad=True)
            # train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_loss = loss.mean().asscalar() / batch_size
            metric_acc.update(label, outputs)
            _, train_acc = metric_acc.get()
            # save parameters
            if i % 100 == 0 and i != 0:
                logger.info("Save parameters")
                net.save_parameters(os.path.join(pretrained_path, 'eco_net_iter_{}.params'.format(str(i))))
            logger.info('[Epoch %d] Iter: %d, Train-acc: %.3f, loss: %.3f | time: %.1f' % (epoch, i, train_acc, train_loss, time.time() - batch_start))
            
        _, train_acc = metric_acc.get()
        train_loss /= num_batch

        _, val_acc = test(net, val_data, ctx)

        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' % (epoch, train_acc, train_loss, val_acc, time.time() - epoch_start))

    # _, test_acc = test(net, test_data, ctx)
    # print('[Finished] Test-acc: %.3f' % (test_acc))



def check_and_print(args):
    """ Check and Print all parameters. """
    # train_path
    print('train_path : %s' % args.train_path)
    # val_path
    print('val_path : %s' % args.val_path)
    # anno_file
    print('anno_file : %s' % args.anno_file)
    # pretrained
    print('pretrained : %s' % args.pretrained)
    # pretrained_path
    print('pretrained_path : %s' % args.pretrained_path)
    # gpus
    print('gpus : %s' % args.gpus)
    # epochs
    print('epochs : %s' % args.epochs)
    # batch_size
    print('batch_size : %d' % args.batch_size)
    # learning_rate
    print('learning_rate : %f' % args.learning_rate)
    # weight_decay
    print('weight_decay : %f' % args.weight_decay)
    # optimizer
    print('optimizer : %s' % args.optimizer)
    # momentum
    print('momentum : %s' % args.momentum)
    # lr_refactor_steps
    print('lr_refactor_steps : %s' % args.lr_refactor_steps)
    # lr_refactor_ratio
    print('lr_refactor_ratio : %f' % args.lr_refactor_ratio)
    # log_file
    print('log_file : %s' % args.log_file)
    # num_classes
    print('num_class : %d' % args.num_class)
    # tensorboard
    print('tensorboard : {}'.format(args.tensorboard))
    # num_workers
    print('num_workers : %d' % args.num_workers)
    # per_device_batch_size
    print('per_device_batch_size : %d' % args.per_device_batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ECO network with pre-train model or from scratch')
    parser.add_argument('--train-path', dest='train_path', default='/data/jh/notebooks/hudengjun/meitu/ECOFrames/', type=str,
                        help='path to training data')
    parser.add_argument('--val-path', dest='val_path', default='/data/jh/notebooks/hudengjun/meitu/ECOFrames/', type=str,
                        help='path to validation data')
    parser.add_argument('--anno-file', dest='anno_file', default='/data/jh/notebooks/hudengjun/meitu/DatasetLabels/short_video_trainingset_annotations.txt.082902', type=str,
                        help='path to annotation file')
    parser.add_argument('--pretrained', dest='pretrained', default=False, type=bool,
                        help='whether use pretrained model')
    parser.add_argument('--pretrained-path', dest='pretrained_path', default='/home/lijie/ECO_Full_kinetics_pretrained/model/', type=str,
                        help='path to pretrained model files')
    parser.add_argument('--gpus', dest='gpus', default='1', type=str,
                        help='Network run on GPU or CPU, eg. 0,1')
    parser.add_argument('--epochs', dest='epochs',  default=10, type=int,
                        help='The time of running all of the images.')
    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int,
                        help='training batch size.')
    parser.add_argument('--lr', dest='learning_rate', default=0.1, type=float,
                        help='learning_rate')
    parser.add_argument('--wd', dest='weight_decay', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--optimizer', dest='optimizer', default='sgd', type=str,
                        help='optimizer object function')
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--lr-steps', dest='lr_refactor_steps', default=None, type=str,
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', default=0.1, type=float,
                        help='ratio to refactor learning rate')
    parser.add_argument('--log', dest='log_file', default='train.log', type=str,
                        help='save training log information to file')
    parser.add_argument('--num-class', dest='num_class', default=63, type=int,
                        help='number of classes')
    parser.add_argument('--tensorboard', dest='tensorboard', type=bool, default=False,
                        help='save metrics into tensorboard readable files')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=8,
                        help='number of process to read data')
    parser.add_argument('--per-device-batch-size', dest='per_device_batch_size', default=8, type=int,
                        help='sub-batch size on each device')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # context list
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    # check parameters
    check_and_print(args)
    # start training
    train_net(args.train_path, args.val_path,args.anno_file,
              args.num_class, args.batch_size,
              args.pretrained, args.pretrained_path, args.epochs,
              ctx, args.learning_rate, args.weight_decay,
              args.optimizer, args.momentum,
              args.lr_refactor_steps, args.lr_refactor_ratio,
              args.log_file, args.tensorboard,
              args.num_workers, args.per_device_batch_size)
              # args.jitter_param, args.lighting_param)


if __name__ == '__main__':
    main()

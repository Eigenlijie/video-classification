#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time

date_format = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def clear_saved_params(path):
    """ Get all files from path, and remove part of them """
    filename_list = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)))
    for filename in filename_list[:-3]:
        if os.path.exists(path + filename) and '.params' == os.path.splitext(filename)[1]:
            os.remove(path + filename)
            print(date_format + " : " + "{} is removed.".format(filename))
        else:
            print("No such file exist.")


if __name__ == '__main__':
    MODEL_PATH = './model/'
    clear_saved_params(MODEL_PATH)


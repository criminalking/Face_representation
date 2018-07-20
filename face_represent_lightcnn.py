'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

from src.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from src.util import print_result, read_list

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--img_list', default='', type=str, metavar='PATH', 
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--save_path', default='', type=str, metavar='PATH', 
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=79077, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')


def main():
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    img_list    = read_list(args.img_list)
    transform   = transforms.Compose([transforms.ToTensor()])
    count       = 0
    input       = torch.zeros(1, 1, 128, 128)
    featurelist = []
    for img_name in img_list:
        count = count + 1
        img   = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        assert (img.shape[0] == img.shape[1]), "You need crop! (%d %d)\n" % (img.shape[0], img.shape[1])
        img   = cv2.resize(img, (128, 128))
        img   = np.reshape(img, (128, 128, 1))
        img   = transform(img)
        input[0,:,:,:] = img

        start = time.time()
        if args.cuda:
            input = input.cuda()
        input_var   = torch.autograd.Variable(input, volatile=True)
        _, features = model(input_var)
        end         = time.time() - start
        print("{}({}/{}). Time: {}".format(img_name, count, len(img_list), end))
        #save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])
        featurelist.append(features.data.cpu().numpy()[0])

    print_result(img_list, featurelist)


def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid   = open(fname, 'wb')
    fid.write(features) # 256 dims
    fid.close()
    

if __name__ == '__main__':
    main()

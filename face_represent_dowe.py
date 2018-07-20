from __future__ import print_function

import numpy as np
from PIL import Image

import caffe
import argparse

from src.util import read_list, print_result


def read_image(filename):
    im = Image.open(filename)
    assert (im.size[0] == im.size[1]), "You need crop!\n"
    im = im.resize((224, 224))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ = in_ / 255.0
    in_ = (in_ - np.array((0.485,0.456,0.406))) / np.array((0.229,0.224,0.225))
    in_ = in_.transpose((2,0,1)) # C x H x W
    return in_


def get_feature(net, in_):
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['pool5'].data[0].reshape(-1) # 2048 dim feature
    return out.copy()


def main(args):
    # init
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # load net
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    img_list = read_list(args.img_list)
    features = [] #len(img_list)
    for img_name in img_list:
        in_ = read_image(img_name)
        features.append(get_feature(net, in_))

    print_result(img_list, features)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Face representation')
    parser.add_argument('--img_list', default='input/list.txt', type=str,
                        help='list of face images')
    parser.add_argument('--prototxt',
                        default='model/ResNet-101-deploy_augmentation.prototxt',
                        type=str, help='path to prototxt')
    parser.add_argument('--caffemodel',
                        default='model/snap_resnet__iter_120000.caffemodel',
                        type=str, help='path to caffemodel')
    args = parser.parse_args()
    main(args)


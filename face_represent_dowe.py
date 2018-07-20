from __future__ import print_function

import numpy as np
from PIL import Image
from numpy import linalg as LA

import caffe
import argparse


def read_list(filename):
    img_list = []
    with open(filename, 'r') as f:
        for line in f.readlines()[0:]:
            if line[0] == '#':
                continue
            img_path = line.strip().split()
            img_list.append(img_path[0])
    return img_list


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


def compute_cos_similarity(feature1, feature2):
    try:
        similarity = np.dot(feature1,feature2) \
                     / (LA.norm(feature1) * LA.norm(feature2))
    except ZeroDivisionError:
        print("Zero division error here!\n")
    return similarity


def compute_l2_distance(feature1, feature2):
    feature1 = feature1.reshape(-1)
    feature2 = feature2.reshape(-1)
    distance = np.sum((feature1 - feature2)**2)
    return distance


def print_result(img_list, features):
    num_images = len(img_list)
    print ('Images:')
    for i in range(num_images):
        print ('%1d: %s' % (i, img_list[i]))
    print ('')

    # Print distance matrix
    print ('Distance matrix')
    print ('    ', end='')
    for i in range(num_images):
        print ('      %1d     ' % i, end='')
    print ('')
    for i in range(num_images):
        print ('%1d  ' % i, end='')
        for j in range(num_images):
            distance = compute_l2_distance(features[i], features[j])
            print ('  %8.4f  ' % distance, end='')
        print ('')

    # Print similarity matrix
    print ('')
    print ('Similarity matrix')
    print ('    ', end='')
    for i in range(num_images):
        print ('    %1d     ' % i, end='')
    print ('')
    for i in range(num_images):
        print ('%1d  ' % i, end='')
        for j in range(num_images):
            similarity = compute_cos_similarity(features[i], features[j])
            print ('  %1.4f  ' % similarity, end='')
        print ('')


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


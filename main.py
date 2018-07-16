import numpy as np
from PIL import Image
from numpy import linalg as LA
import matplotlib.pyplot as plt

import caffe

def read_image(filename):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(filename)
    #im = im.crop((95, 120, 365, 390)) # x0, y0, x1, y1, for 04250
    im = im.resize((224, 224))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ = in_ / 255.0
    in_ = (in_ - np.array((0.485,0.456,0.406))) / np.array((0.229,0.224,0.225))
    in_ = in_.transpose((2,0,1))
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

# init
caffe.set_device(0)
caffe.set_mode_gpu()
# load net
net = caffe.Net('model/ResNet-101-deploy_augmentation.prototxt',
                'model/snap_resnet__iter_120000.caffemodel', caffe.TEST)

in1_ = read_image('input/image_right.jpg')
feature1 = get_feature(net, in1_)
in2_ = read_image('input/image_left.jpg')
feature2 = get_feature(net, in2_)

print feature1, feature2

similarity = compute_cos_similarity(feature1, feature2)
print "Similarity: ", similarity

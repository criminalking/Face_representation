import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import misc
import os
import time
import argparse

from src.facenet_pytorch_model import KitModel
from src.util import *

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def main(args):
    model = KitModel(args.model).cuda()
    model.eval()

    image_size = args.image_size
    image_list = read_list(args.image_list)

    images = torch.zeros((len(image_list), 3, image_size, image_size))
    for i, image in enumerate(image_list):
        img = misc.imread(os.path.expanduser(image), mode='RGB') # N x H x W x C
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')
        img = prewhiten(img)

        img = img.transpose((2, 0, 1))
        img = img.astype('float32')

        img = torch.from_numpy(img)
        images[i] = img

    start = time.time()
    with torch.no_grad():
        input = torch.autograd.Variable(images.cuda())
        emb = model(input)
    end = time.time() - start

    print_result(image_list, emb.cpu().numpy())
    print(end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/facenet_pytorch.npy',
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_list', type=str, default='input/list.txt', help='Image list to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    args = parser.parse_args()
    main(args)

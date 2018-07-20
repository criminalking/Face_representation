from __future__ import print_function
import numpy as np
from numpy import linalg as LA


def read_list(filename):
    img_list = []
    with open(filename, 'r') as f:
        for line in f.readlines()[0:]:
            if line[0] == '#':
                continue
            img_path = line.strip().split()
            img_list.append(img_path[0])
    return img_list


def compute_cos_similarity(feature1, feature2):
    feature1 = feature1.reshape(-1)
    feature2 = feature2.reshape(-1)
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
    """Print similarity and distance result"""
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

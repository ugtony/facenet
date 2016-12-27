"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import cv2

def main(args):
  
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_growth = True, log_device_placement=False))
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        with sess.as_default():
            if args.device:
                with tf.device(args.device):
                    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, '../../data/')
            else:
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, '../../data/')
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    try:
        img = misc.imread(args.image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))
            exit()
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        if img.shape[2] < 3:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))
            exit()
        img = img[:,:,0:3]
        #check image size, don't be too big
        resize_ratio = 1.0
        if max(img.shape[0:2]) > args.maximum_image_size:
            resize_ratio = args.maximum_image_size / max(img.shape[0:2])
            img = misc.imresize(img, resize_ratio)

        bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        for i in range(0,5):
            cv2.circle(img, (points[i], points[5+i]), 3, (0,0,255))
        cv2.imwrite("out.jpg", img)
        print(bounding_boxes, points)
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type=str, help='Directory with unaligned images.')        
    parser.add_argument('--device', type=str,
        help='Assign CPU/GPU. e.g., "/cpu:0", "/gpu:1"', default=None)    
    parser.add_argument('--maximum_image_size', type=int,
        help='If the image is larger than this size, its maximum dimension will be shrinked to this first', default=2048)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

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
import facenet
import align.detect_face
import random

def main(args):
  
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
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

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)    
        
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)                
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        if img.shape[2] < 3:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        img = img[:,:,0:3]
                        #check image size, don't be too big
                        resize_ratio = 1.0
                        if max(img.shape[0:2]) > args.maximum_image_size:
                            resize_ratio = args.maximum_image_size / max(img.shape[0:2])
                            img = misc.imresize(img, resize_ratio)
    
                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            
                            meta = readMeta(args.meta_dir, cls.name, image_path[image_path.rfind('/')+1:image_path.rfind('.')])
                            bbox = meta['bbox']
                            bbox = bbox * resize_ratio
                            
                            #todo: choose the bbox with maximum overlap(intersection/union) ratio
                            if nrof_faces>1:
                                maxleft = np.maximum(bbox[0], det[:,0])
                                minleft = np.minimum(bbox[0], det[:,0])
                                maxtop = np.maximum(bbox[1], det[:,1])
                                mintop = np.minimum(bbox[1], det[:,1])
                                maxright = np.maximum(bbox[2], det[:,2])                                
                                minright = np.minimum(bbox[2], det[:,2])                                
                                maxdown = np.maximum(bbox[3], det[:,3])
                                mindown = np.minimum(bbox[3], det[:,3])
                                
                                intersection = np.maximum(0,minright-maxleft+1)*np.maximum(0,mindown-maxtop+1)
                                union = (maxright-minleft+1)*(maxdown-mintop+1)
                                overlap = intersection/union                                
                                index = np.argmax(overlap)
                                
                                #bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                #img_center = img_size / 2
                                #offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                #offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                #index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det = det[index,:]
                                                                
                                
                                
                            det = np.squeeze(det)
                            
                            #check det
                            maxleft = np.maximum(bbox[0], det[0])
                            minleft = np.minimum(bbox[0], det[0])
                            maxtop = np.maximum(bbox[1], det[1])
                            mintop = np.minimum(bbox[1], det[1])
                            maxright = np.maximum(bbox[2], det[2])                                
                            minright = np.minimum(bbox[2], det[2])                                
                            maxdown = np.maximum(bbox[3], det[3])
                            mindown = np.minimum(bbox[3], det[3])
                                
                            intersection = np.maximum(0,minright-maxleft+1)*np.maximum(0,mindown-maxtop+1)
                            union = (maxright-minleft+1)*(maxdown-mintop+1)
                            overlap = intersection/union
                            if overlap < args.overlap_threshold:
                                print('Face detected "%s" with bad positioin' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            
                            
                            
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-args.margin/2, 0)
                            bb[1] = np.maximum(det[1]-args.margin/2, 0)
                            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    
def readMeta(dirpath, name, image_id):
    with open(os.path.join(dirpath, name+'.txt'), "r") as f:
        for line in f:
            fields = line.split()
            if fields[0] == image_id:
                bbox = np.array(fields[2:6], np.float32) #[left top right bottom]
                return {'bbox': bbox}
        return None
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('meta_dir', type=str, help='Directory with Vggface meta')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--device', type=str,
        help='Assign CPU/GPU. e.g., "/cpu:0", "/gpu:1"', default=None)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--overlap_threshold', type=float,
        help='Lowest overlap threshold, if the overlaped area among detected face and the VGGFace meta is lower than this, it will be identified as a failed detection.', default=0.3)
    parser.add_argument('--maximum_image_size', type=int,
        help='If the image is larger than this size, its maximum dimension will be shrinked to this first', default=2048)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

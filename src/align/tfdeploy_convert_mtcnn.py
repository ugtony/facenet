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

import tfdeploy as td
import tensorflow as tf
import numpy as np
import align.detect_face
import sys
import os

model_path = '../../data/'
def main(args):
  
	td.setup(tf, td.IMPL_SCIPY)
	#td.setup(tf)
	
	print('Creating networks and loading parameters')
	
	with tf.Graph().as_default():		
		sess = tf.Session()		
		with sess.as_default():
			with tf.variable_scope('pnet'):
				data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
				pnet = align.detect_face.PNet({'data':data})
				pnet.load(os.path.join(model_path, 'det1.npy'), sess)
			with tf.variable_scope('rnet'):
				data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
				rnet = align.detect_face.RNet({'data':data})
				rnet.load(os.path.join(model_path, 'det2.npy'), sess)
			with tf.variable_scope('onet'):
				data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
				onet = align.detect_face.ONet({'data':data})
				onet.load(os.path.join(model_path, 'det3.npy'), sess)

			pnet_model = td.Model()
			pnet_model.add(tf.get_default_graph().get_tensor_by_name('pnet/conv4-2/BiasAdd:0'), sess)
			pnet_model.add(tf.get_default_graph().get_tensor_by_name('pnet/prob1:0'), sess)			
			pnet_model.save(os.path.join(model_path, "pnet.pkl"))
			rnet_model = td.Model()
			rnet_model.add(tf.get_default_graph().get_tensor_by_name('rnet/conv5-2/conv5-2:0'), sess)
			rnet_model.add(tf.get_default_graph().get_tensor_by_name('rnet/prob1:0'), sess)
			rnet_model.save(os.path.join(model_path, "rnet.pkl"))
			onet_model = td.Model()
			onet_model.add(tf.get_default_graph().get_tensor_by_name('onet/conv6-2/conv6-2:0'), sess)
			onet_model.add(tf.get_default_graph().get_tensor_by_name('onet/conv6-3/conv6-3:0'), sess)
			onet_model.add(tf.get_default_graph().get_tensor_by_name('onet/prob1:0'), sess)
			onet_model.save(os.path.join(model_path, "onet.pkl"))
			
if __name__ == '__main__':
	main(sys.argv[1:])

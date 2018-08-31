"""
This source code was modified from this resource:
https://github.com/argman/EAST
We download pre-trained model and customize several function

Please refer to that link if you want to copy or modify EAST text detector
"""


import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import model

tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('min_confidence', '0.5', '')

FLAGS = tf.app.flags.FLAGS

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = max(resize_h, resize_w)
    resize_w = resize_h

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32

    top 	= int((resize_h - h) / 2)
    bottom 	= resize_h - (h + top)
    left	= int((resize_w - w) / 2)
    right	= resize_w - (w + left)

    color = [0, 0, 0]
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=color)

    (h, w) = im.shape[:2]
    print('new size : ' + str(w) + ' x ' + str(h))
    return im, (1, 1)

def isText(im):
	returnVal = False
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

	data = []

	with tf.get_default_graph().as_default():
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

		f_score, f_geometry = model.model(input_images, is_training=False)

		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
		saver = tf.train.Saver(variable_averages.variables_to_restore())

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
			model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
			print('Restore from {}'.format(model_path))
			saver.restore(sess, model_path)

			im = im[:, :, ::-1]
			start_time = time.time()
			im_resized, (ratio_h, ratio_w) = resize_image(im)
			im = im_resized[:, :, ::-1]

			timer = {'net': 0, 'restore': 0, 'nms': 0}
			start = time.time()
			score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
			timer['net'] = time.time() - start
			
			max_score = np.max(score)

			if(float(max_score) >= float(FLAGS.min_confidence)):
				returnVal = True
				

			print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
				'image', timer['net']*1000, timer['restore']*1000, timer['nms']*1000))


			duration = time.time() - start_time
			print('[timing] {}'.format(duration))
	
			return returnVal

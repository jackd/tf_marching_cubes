#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_marching_cubes.wrapped import find_contours
import matplotlib.pyplot as plt


n = 32
x = np.linspace(-1, 1, n)
xy = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
# centres = np.array([[0, 0], [-0.5, -0.5], [0.5, 0.5]])
centres = np.array([[-0.5, -0.5], [0.5, 0.5]])
# centres = np.array([[0.5, 0.5]])
# centres = np.array([[-0.5, -0.5]])
# # c0 = centres[0]
# print
# print((np.expand_dims(xy, axis=-2)-c0).shape)
# exit()
# print(np.sum((np.expand_dims(xy, axis=-2) - c0)**2, axis=-1).shape)
# exit()
level = 0.1

data = np.sqrt(np.min(
    np.sum((np.expand_dims(xy, axis=-2) - centres)**2, axis=-1), axis=-1))

# contours = measure.find_contours(data, level)
verts, lengths = find_contours(
    tf.constant(data, dtype=tf.float32), level, back_prop=True)

with tf.Session() as sess:
    verts, lengths = sess.run((verts, lengths))
contours = np.split(verts, lengths)

# contours = [verts[starts[i]: starts[i+1]] for i in range(len(starts) - 1)]

plt.imshow(data)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0])
plt.show()

#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mayavi.mlab as mlab
from skimage.draw import ellipsoid
import tensorflow as tf
from tf_marching_cubes import isosurface

# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(16, 20, 16, levelset=True)
ellip_double = np.concatenate((ellip_base[:-1, ...],
                               ellip_base[2:, ...]), axis=0).astype(np.float32)

ellip_double = tf.constant(np.array(ellip_double), dtype=tf.float32)

# Use marching cubes to obtain surface meshes at different levels
v0, f0 = isosurface(ellip_double, 0.2)
v1, f1 = isosurface(ellip_double, 0.5)
with tf.Session() as sess:
    v0, f0, v1, f1 = sess.run((v0, f0, v1, f1))

figure = mlab.figure()

for verts, faces, color in ((v0, f0, (0, 0, 1)), (v1, f1, (0, 1, 0))):
    x, y, z = verts.T
    mlab.triangular_mesh(
        x, y, z, faces, figure=figure, color=color, opacity=0.3)
    mlab.triangular_mesh(x, y, z, faces, representation='wireframe',
                         color=(0, 0, 0), figure=figure, opacity=0.2)
mlab.show()

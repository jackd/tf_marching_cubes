#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
import numpy as np
import tensorflow as tf
import mayavi.mlab as mlab
from tf_marching_cubes import isosurface
import tf_marching_cubes.wrapped as wrapped
from skimage.draw import ellipsoid

# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(16, 20, 16, levelset=True)
data = np.concatenate((ellip_base[:-1, ...],
                      ellip_base[2:, ...]), axis=0).astype(np.float32)

data = tf.constant(data, dtype=tf.float32)
level = 0.1

# Use marching cubes to obtain surface meshes at different levels
v, f = isosurface(data, level)
v_wrapped, f_wrapped = wrapped.marching_cubes_lewiner(data, level=level)[:2]
# v_wrapped, f_wrapped = wrapped.marching_cubes_classic(data, level=level)[:2]
v_hacked = wrapped.vertex_gradient_hack(v_wrapped, data, level)
n_warm_up = 5
n_runs = 5
with tf.Session() as sess:
    for i in range(n_warm_up):
        sess.run((v, f, v_wrapped, f_wrapped))
    t = time()
    for i in range(n_runs):
        v0, f0 = sess.run((v, f))
    dt = time() - t
    t = time()
    for i in range(n_runs):
        v1, f1 = sess.run((v_wrapped, f_wrapped))
    dt_wrapped = time() - t
    t = time()
    for i in range(n_runs):
        v2, f2 = sess.run((v_hacked, f_wrapped))
    dt_hacked = time() - t

# print(v1[:5])
# print('---')
# print(v2[:5])
# exit()

print('isosurface dt: %.3f' % dt)
print('wrapped dt: %.3f' % dt_wrapped)
print('hacked dt: %.3f' % dt_hacked)

for verts, faces, color in (
            (v0, f0, (0, 0, 1)),
            (v1, f1, (0, 1, 0)),
            (v2, f2, (1, 0, 0))
        ):
    mlab.figure()
    x, y, z = verts.T
    mlab.triangular_mesh(
        x, y, z, faces, color=color, opacity=0.3)
    mlab.triangular_mesh(x, y, z, faces, representation='wireframe',
                         color=(0, 0, 0), opacity=0.2)
mlab.show()

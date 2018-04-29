from __future__ import division
try:
    from progress.bar import IncrementalBar
except ImportError:
    pass
import numpy as np
import tensorflow as tf
from tf_marching_cubes import isosurface

n = 32
x = np.linspace(-2, 2, n)
x, y, z = np.meshgrid(x, x, x)
x = 1 - (x*x + y*y + z)

x = tf.Variable(x.astype(np.float32))
x = tf.pad(x, [[1, 1], [1, 1], [1, 1]], constant_values=-1)

verts, faces = isosurface(x, 0)
verts = (verts - n/2) * (4 / n)

radius = tf.reduce_sum(verts**2, axis=-1)
loss = tf.reduce_sum((radius - 1)**2)

opt = tf.train.AdamOptimizer(1e-1).minimize(loss)

n1 = 1000
n2 = 3


def vis_mesh(vertices, faces, include_wireframe=True, color=(0, 0, 1),
             **kwargs):
    from mayavi import mlab
    if len(faces) == 0:
        print('Warning: no faces')
        return
    x, y, z = vertices.T
    mlab.triangular_mesh(x, y, z, faces, color=color, **kwargs)
    if include_wireframe:
        mlab.triangular_mesh(
            x, y, z, faces, color=(0, 0, 0), representation='wireframe')
    mlab.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    v, f = sess.run([verts, faces])
    vis_mesh(v, f)
    for _ in range(n2):

        try:
            bar = IncrementalBar(max=n1)
        except NameError:
            bar = None
        for i in range(n1):
            if bar is not None:
                bar.next()
            sess.run(opt)
            # print(loss_val)
        v, f, loss_val = sess.run([verts, faces, loss])
        vis_mesh(v, f)
        print(loss_val)
        if bar is not None:
            bar.finish()

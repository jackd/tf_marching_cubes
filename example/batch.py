import tensorflow as tf
import numpy as np
import os
from tf_marching_cubes import batch_padded_isosurface

folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data')
fns = ['car_vox.npy', 'plane_vox.npy']
voxels = np.array(
    [np.load(os.path.join(folder, fn)) for fn in fns], dtype=np.bool)

data = np.empty(shape=voxels.shape, dtype=np.float32)
data[voxels] = 1
data[np.logical_not(voxels)] = -1

max_vertices = 5000
max_faces = 5000

verts, faces, nv, nf = batch_padded_isosurface(
    data, 0, max_vertices, max_faces)


def vis_meshes(verts, faces, num_verts, num_faces):
    from mayavi import mlab
    for v, f, nv, nf in zip(verts, faces, num_verts, num_faces):
        v = v[:nv]
        f = f[:nf]
        mlab.figure()
        x, y, z = v.T
        mlab.triangular_mesh(x, y, z, f, color=(0, 0, 1))
        mlab.triangular_mesh(x, y, z, f, representation='wireframe',
                             color=(0, 0, 0))
    mlab.show()


with tf.Session() as sess:
    v, f, nv, nf = sess.run((verts, faces, nv, nf))


vis_meshes(v, f, nv, nf)

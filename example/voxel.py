import os
import numpy as np
import tensorflow as tf
from tf_marching_cubes import isosurface


voxel_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data', 'plane_vox.npy')


voxels = np.load(voxel_path)
data = np.empty(shape=voxels.shape, dtype=np.float32)
data[voxels] = 1
data[np.logical_not(voxels)] = -1

verts, faces = isosurface(data, 0)
with tf.Session() as sess:
    v, f = sess.run([verts, faces])


def vis(v, f, voxels):
    from mayavi import mlab

    def vis_voxels(voxels, **kwargs):
        x, y, z = np.where(voxels)
        if len(x) == 0:
            Warning('No voxels to display')
        else:
            if 'mode' not in kwargs:
                kwargs['mode'] = 'cube'
            mlab.points3d(x, y, z, **kwargs)

    def vis_mesh(vertices, faces, include_wireframe=True, color=(0, 0, 1),
                 **kwargs):
        if len(faces) == 0:
            print('Warning: no faces')
            return
        x, y, z = vertices.T
        mlab.triangular_mesh(x, y, z, faces, color=color, **kwargs)
        if include_wireframe:
            mlab.triangular_mesh(
                x, y, z, faces, color=(0, 0, 0), representation='wireframe')

    mlab.figure()
    vis_voxels(voxels, color=(0, 0, 1))
    mlab.figure()
    vis_mesh(v, f, color=(0, 1, 0))
    mlab.show()


vis(v, f, voxels)

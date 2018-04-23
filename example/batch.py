import tensorflow as tf
import numpy as np
import os
from shapenet.core import cat_desc_to_id
from shapenet.core.voxels import VoxelConfig
from tf_marching_cubes import isosurface

cat_desc = 'car'
cat_id = cat_desc_to_id(cat_desc)
config = VoxelConfig()
with config.get_dataset(cat_id) as ds:
    key = tuple(ds.keys())[0]
    voxels = ds[key].dense_data(fix_coords=False)

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


def map_fn(x):
    verts, faces = isosurface(x, 0)
    verts = tf.pad(
        verts, [[0, max_vertices - tf.shape(verts)[0]], [0, 0]],
        constant_values=np.inf)
    faces = tf.pad(
        faces, [[0, max_faces - tf.shape(faces)[0]], [0, 0]],
        constant_values=-1)
    return verts, faces


verts, faces = tf.map_fn(map_fn, data, dtype=(tf.float32, tf.int32))


def vis_meshes(verts, faces):
    from mayavi import mlab
    for v, f in zip(verts, faces):
        mlab.figure()
        f = f[np.all(f != -1, axis=-1)]
        v = v[np.all(np.isfinite(v), axis=-1)]
        x, y, z = v.T
        mlab.triangular_mesh(x, y, z, f, color=(0, 0, 1))
        mlab.triangular_mesh(x, y, z, f, representation='wireframe',
                             color=(0, 0, 0))
    mlab.show()


with tf.Session() as sess:
    v, f = sess.run((verts, faces))

vis_meshes(v, f)

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mayavi.mlab as mlab

# from skimage import measure
# from tf_marching_cubes.np_impl import isosurface
from tf_marching_cubes import isosurface
from skimage.draw import ellipsoid
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()


# Generate a level set about zero of two identical ellipsoids in 3D
ellip_base = ellipsoid(16, 20, 16, levelset=True)
ellip_double = np.concatenate((ellip_base[:-1, ...],
                               ellip_base[2:, ...]), axis=0).astype(np.float32)

ellip_double = tf.constant(np.array(ellip_double), dtype=tf.float32)

# Use marching cubes to obtain the surface mesh of these ellipsoids
# verts, faces, normals, values = measure.marching_cubes_lewiner(
#                      ellip_double, 0)
verts, faces = isosurface(ellip_double, 0.2)
with tf.Session() as sess:
    verts, faces = sess.run((verts, faces))

x, y, z = (verts[:, i] for i in range(3))
figure = mlab.figure()
# mlab.points3d(x, y, z, figure=figure)
mlab.triangular_mesh(x, y, z, faces, figure=figure, color=(0, 0, 1))
mlab.triangular_mesh(x, y, z, faces, representation='wireframe',
                     color=(0, 0, 0), figure=figure)
mlab.show()

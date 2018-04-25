"""Provides a tf wrapper for skimage marching cubes implementations."""
import numpy as np
import tensorflow as tf
from skimage import measure


def find_contours(data, level, back_prop=False, **kwargs):
    """
    Tensorflow wrapper around `skimage.measure.find_contours`.

    Args:
        data: 2D array with embedding function values.
        level: value of isosurface to extract
        back_prop: if True, gradients can propagate through vertices via
            `vertex_gradient_hack`.
        **kwargs: passed to `skimage.measure.find_contours`.

    Returns:
        vertices: (N, 2) array of vertices. Duplicates indicate closed loops.
            Concatenated contours from `find_contours`.
        lengths: (m,) array of contour lengths.
    """
    def fn(data):
        contours = measure.find_contours(data, level, **kwargs)
        verts = np.concatenate(contours, axis=0)
        lengths = np.array([len(c) for c in contours], dtype=np.int32)
        return verts, lengths

    with tf.name_scope('find_contours'):
        verts, lengths = tf.py_func(
            fn, (data,), (tf.float32, tf.int32), stateful=False)
        verts.set_shape((None, 2))
        lengths.set_shape((None,))
        if back_prop:
            verts = vertex_gradient_hack(verts, data, level=level)
    return verts, lengths


def marching_cubes_classic(volume, level, back_prop=False, **kwargs):
    """
    Tensorflow wrapper around `skimage.measure.marching_cubes_classic`.

    Args:
        volume: embedding function data evaluated on a regular grid.
        level: value of isosurface to extract
        back_prop: if True, gradients can propagate through vertices via
            `vertex_gradient_hack`
        **kwargs: passed to `skimage.measure.marching_cubes_classic`

    Note: the outputs are not differentiable.

    See skimage.measure.marching_cubes_classic for *args, **kwargs details.
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_classic
    """
    def fn(vol):
        vertices, faces = measure.marching_cubes_classic(
            vol, level, **kwargs)
        return vertices.astype(np.float32), faces.astype(np.int32)

    with tf.name_scope('marching_cubes_classic'):
        verts, faces = tf.py_func(
            fn, (volume,), (tf.float32, tf.int32), stateful=False)
        verts.set_shape((None, 3))
        faces.set_shape((None, 3))
        if back_prop:
            if 'spacing' in kwargs and kwargs['spacing'] != (1, 1, 1):
                raise NotImplementedError(
                    'Non-unit spacing not supported')
            verts = vertex_gradient_hack(verts, volume, level=level)
    return verts, faces


def get_normals(verts, faces, normalize=False):
    vf = tf.gather(verts, faces)
    u, v, w = tf.unstack(vf, axis=-1)
    normals = tf.cross(v - u, w - u)
    if normalize:
        normals = normals / tf.sqrt(
            tf.reduce_sum(normals**2, axis=-1, keepdims=True))
    return normals


def marching_cubes_lewiner(
        volume, level, back_prop=False, back_prop_normals=False, **kwargs):
    """
    Tensorflow wrapper around `skimage.meaure.marching_cubes_lewiner`.

    Args:
        volume: 3D volumetric tensor of embedding values
        level: value of isosurface to extract
        back_prop: if True, allows gradient to propagate through vertices via
            `vertex_gradient_hack`. Does not allow propagation through normals.
        back_prop_normals: if True, allows gradient to propagate through
            normals by recalculating them based on vertices and faces.
            Raises an error is `back_prop` is not also True.
        *args, **kwargs: passed to wrapped function. Must be normal python
            variables, not tensors.

    Returns:
        tensors:
            (V, 3) float32 verts tensor
            (F, 3) int32 face tensor
            (V, 3) float32 normals tensor
            (V,) float32 values tensor

    See skimage.measure.marching_cubes_lewiner for more.
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    """
    def fn(vol):
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            vol, level=level, **kwargs)
        return verts, faces, normals, values

    with tf.name_scope('marching_cubes_lewiner'):
        verts, faces, normals, values = tf.py_func(
            fn, (volume,), (tf.float32, tf.int32, tf.float32, tf.float32),
            stateful=False)
        verts.set_shape((None, 3))
        faces.set_shape((None, 3))
        normals.set_shape((None, 3))
        values.set_shape((None,))
        if back_prop:
            if 'spacing' in kwargs and kwargs['spacing'] != (1, 1, 1):
                raise NotImplementedError(
                    'Non-unit spacing not supported')
            verts = vertex_gradient_hack(verts, volume, level=level)
        if back_prop_normals:
            if not back_prop:
                raise ValueError('back_prop must be true back_prop_normals is')
            normals = get_normals(verts, faces, normalize=True)
    return verts, faces, normals, values


def vertex_gradient_hack(vertices, data, level=0):
    """
    Get vertices at interpolated roots of data embedding fn with gradients.

    Args:
        vertices: (N, ndims) float32 array of vertex values from linearly
            interpolated embedding function values
        data: rank `ndims` embedding function values used to generate vertices

    Returns:
        (N, ndims) float32 array of vertex values with gradient information.
    """
    with tf.name_scope('vertex_gradient_hack'):
        v0 = tf.floor(vertices)
        v0i = tf.cast(v0, tf.int32)
        v1 = tf.ceil(vertices)
        v1i = tf.cast(v1, tf.int32)

        f0 = tf.gather_nd(data, v0i)
        f1 = tf.gather_nd(data, v1i)

        numer = f0 if level == 0 else f0 - level
        alpha = tf.expand_dims(numer / (f0 - f1), axis=-1)
        interped = (1 - alpha)*v0 + alpha*v1

    return interped
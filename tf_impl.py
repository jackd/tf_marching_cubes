from __future__ import division
import tensorflow as tf
import numpy as np


def _gathered(indices, shape, fn):
    values = tf.ones(shape=tf.shape(indices)[0], dtype=tf.int32)
    condition = tf.scatter_nd(indices, values, shape)
    return fn(condition)


def _scatter_op1d(ref, condition, true_transform, false_transform=None):
    condition = tf.cast(condition, tf.int32)
    partitioned_data = tf.dynamic_partition(ref, condition, 2)
    if false_transform is not None:
        partitioned_data[0] = false_transform(partitioned_data[0])
    partitioned_data[1] = true_transform(partitioned_data[1])
    condition_indices = tf.dynamic_partition(
        tf.range(tf.shape(ref)[0]), condition, 2)
    return tf.dynamic_stitch(condition_indices, partitioned_data)


def _scatter_op(ref, condition, true_transform, false_transform=None):
    shape = condition.shape
    if len(shape) == 1:
        return _scatter_op1d(
            ref, condition, true_transform, false_transform)
    else:
        ref_shape = ref.shape.as_list()
        ref_shape = [-1 if s is None else s for s in ref_shape]
        new_shape = tf.concat(
            [tf.constant([-1], dtype=tf.int32), ref_shape[len(shape)+1:]],
            axis=0)
        ref = tf.reshape(
            ref, new_shape)
        condition = tf.reshape(condition, (-1,))
        updated = _scatter_op1d(
            ref, condition, true_transform, false_transform)
        return tf.reshape(updated, ref_shape)


def scatter_added(ref, condition, updates):
    """
    Performs scatter_add for non-variables, i.e.
        ref[mask] += updates.
    """
    with tf.name_scope('scatter_added'):
        return _scatter_op(ref, condition, lambda x: x + updates)


def gather_added(ref, indices, updates):
    def f(condition):
        return scatter_added(ref, condition, updates)

    return _gathered(indices, ref.shape, f)


def scatter_updated(ref, condition, updates):
    with tf.name_scope('scatter_updated'):
        return _scatter_op(ref, condition, lambda x: updates)


def gather_updated(ref, indices, updates):
    def f(condition):
        return scatter_updated(ref, condition, updates)

    return _gathered(indices, ref.shape, f)


def _get_isosurface_data():
    # map from grid cell index to edge index.
    # grid cell index tells us which corners are below the isosurface,
    # edge index tells us which edges are cut by the isosurface.
    # (Data stolen from Bourk; see above.)
    edgeTable = np.array([
          0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,  # NOQA
          0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,  # NOQA
          0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,  # NOQA
          0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,  # NOQA
          0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,  # NOQA
          0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,  # NOQA
          0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,  # NOQA
          0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,  # NOQA
          0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,  # NOQA
          0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,  # NOQA
          0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,  # NOQA
          0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,  # NOQA
          0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,  # NOQA
          0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,  # NOQA
          0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,  # NOQA
          0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,  # NOQA
          0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,  # NOQA
          0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,  # NOQA
          0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,  # NOQA
          0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,  # NOQA
          0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,  # NOQA
          0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,  # NOQA
          0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,  # NOQA
          0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,  # NOQA
          0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,  # NOQA
          0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,  # NOQA
          0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,  # NOQA
          0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,  # NOQA
          0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,  # NOQA
          0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,  # NOQA
          0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,  # NOQA
          0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0  # NOQA
          ], dtype=np.uint16)

    # Table of triangles to use for filling each grid cell.
    # Each set of three integers tells us which three edges to
    # draw a triangle between.
    # (Data stolen from Bourk; see above.)
    triTable = [
        [],
        [0, 8, 3],
        [0, 1, 9],
        [1, 8, 3, 9, 8, 1],
        [1, 2, 10],
        [0, 8, 3, 1, 2, 10],
        [9, 2, 10, 0, 2, 9],
        [2, 8, 3, 2, 10, 8, 10, 9, 8],
        [3, 11, 2],
        [0, 11, 2, 8, 11, 0],
        [1, 9, 0, 2, 3, 11],
        [1, 11, 2, 1, 9, 11, 9, 8, 11],
        [3, 10, 1, 11, 10, 3],
        [0, 10, 1, 0, 8, 10, 8, 11, 10],
        [3, 9, 0, 3, 11, 9, 11, 10, 9],
        [9, 8, 10, 10, 8, 11],
        [4, 7, 8],
        [4, 3, 0, 7, 3, 4],
        [0, 1, 9, 8, 4, 7],
        [4, 1, 9, 4, 7, 1, 7, 3, 1],
        [1, 2, 10, 8, 4, 7],
        [3, 4, 7, 3, 0, 4, 1, 2, 10],
        [9, 2, 10, 9, 0, 2, 8, 4, 7],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
        [8, 4, 7, 3, 11, 2],
        [11, 4, 7, 11, 2, 4, 2, 0, 4],
        [9, 0, 1, 8, 4, 7, 2, 3, 11],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
        [4, 7, 11, 4, 11, 9, 9, 11, 10],
        [9, 5, 4],
        [9, 5, 4, 0, 8, 3],
        [0, 5, 4, 1, 5, 0],
        [8, 5, 4, 8, 3, 5, 3, 1, 5],
        [1, 2, 10, 9, 5, 4],
        [3, 0, 8, 1, 2, 10, 4, 9, 5],
        [5, 2, 10, 5, 4, 2, 4, 0, 2],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
        [9, 5, 4, 2, 3, 11],
        [0, 11, 2, 0, 8, 11, 4, 9, 5],
        [0, 5, 4, 0, 1, 5, 2, 3, 11],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
        [10, 3, 11, 10, 1, 3, 9, 5, 4],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
        [5, 4, 8, 5, 8, 10, 10, 8, 11],
        [9, 7, 8, 5, 7, 9],
        [9, 3, 0, 9, 5, 3, 5, 7, 3],
        [0, 7, 8, 0, 1, 7, 1, 5, 7],
        [1, 5, 3, 3, 5, 7],
        [9, 7, 8, 9, 5, 7, 10, 1, 2],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
        [2, 10, 5, 2, 5, 3, 3, 5, 7],
        [7, 9, 5, 7, 8, 9, 3, 11, 2],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
        [11, 2, 1, 11, 1, 7, 7, 1, 5],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
        [11, 10, 5, 7, 11, 5],
        [10, 6, 5],
        [0, 8, 3, 5, 10, 6],
        [9, 0, 1, 5, 10, 6],
        [1, 8, 3, 1, 9, 8, 5, 10, 6],
        [1, 6, 5, 2, 6, 1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8],
        [9, 6, 5, 9, 0, 6, 0, 2, 6],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
        [2, 3, 11, 10, 6, 5],
        [11, 0, 8, 11, 2, 0, 10, 6, 5],
        [0, 1, 9, 2, 3, 11, 5, 10, 6],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
        [6, 3, 11, 6, 5, 3, 5, 1, 3],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
        [6, 5, 9, 6, 9, 11, 11, 9, 8],
        [5, 10, 6, 4, 7, 8],
        [4, 3, 0, 4, 7, 3, 6, 5, 10],
        [1, 9, 0, 5, 10, 6, 8, 4, 7],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
        [6, 1, 2, 6, 5, 1, 4, 7, 8],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
        [3, 11, 2, 7, 8, 4, 10, 6, 5],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
        [10, 4, 9, 6, 4, 10],
        [4, 10, 6, 4, 9, 10, 0, 8, 3],
        [10, 0, 1, 10, 6, 0, 6, 4, 0],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
        [1, 4, 9, 1, 2, 4, 2, 6, 4],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
        [0, 2, 4, 4, 2, 6],
        [8, 3, 2, 8, 2, 4, 4, 2, 6],
        [10, 4, 9, 10, 6, 4, 11, 2, 3],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4],
        [6, 4, 8, 11, 6, 8],
        [7, 10, 6, 7, 8, 10, 8, 9, 10],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
        [10, 6, 7, 10, 7, 1, 1, 7, 3],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
        [7, 8, 0, 7, 0, 6, 6, 0, 2],
        [7, 3, 2, 6, 7, 2],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
        [0, 9, 1, 11, 6, 7],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
        [7, 11, 6],
        [7, 6, 11],
        [3, 0, 8, 11, 7, 6],
        [0, 1, 9, 11, 7, 6],
        [8, 1, 9, 8, 3, 1, 11, 7, 6],
        [10, 1, 2, 6, 11, 7],
        [1, 2, 10, 3, 0, 8, 6, 11, 7],
        [2, 9, 0, 2, 10, 9, 6, 11, 7],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
        [7, 2, 3, 6, 2, 7],
        [7, 0, 8, 7, 6, 0, 6, 2, 0],
        [2, 7, 6, 2, 3, 7, 0, 1, 9],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
        [10, 7, 6, 10, 1, 7, 1, 3, 7],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
        [7, 6, 10, 7, 10, 8, 8, 10, 9],
        [6, 8, 4, 11, 8, 6],
        [3, 6, 11, 3, 0, 6, 0, 4, 6],
        [8, 6, 11, 8, 4, 6, 9, 0, 1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
        [6, 8, 4, 6, 11, 8, 2, 10, 1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
        [8, 2, 3, 8, 4, 2, 4, 6, 2],
        [0, 4, 2, 4, 6, 2],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
        [1, 9, 4, 1, 4, 2, 2, 4, 6],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
        [10, 9, 4, 6, 10, 4],
        [4, 9, 5, 7, 6, 11],
        [0, 8, 3, 4, 9, 5, 11, 7, 6],
        [5, 0, 1, 5, 4, 0, 7, 6, 11],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
        [9, 5, 4, 10, 1, 2, 7, 6, 11],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
        [7, 2, 3, 7, 6, 2, 5, 4, 9],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
        [6, 9, 5, 6, 11, 9, 11, 8, 9],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
        [6, 11, 3, 6, 3, 5, 5, 3, 1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
        [9, 5, 6, 9, 6, 0, 0, 6, 2],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
        [1, 5, 6, 2, 1, 6],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
        [0, 3, 8, 5, 6, 10],
        [10, 5, 6],
        [11, 5, 10, 7, 5, 11],
        [11, 5, 10, 11, 7, 5, 8, 3, 0],
        [5, 11, 7, 5, 10, 11, 1, 9, 0],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
        [2, 5, 10, 2, 3, 5, 3, 7, 5],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
        [1, 3, 5, 3, 7, 5],
        [0, 8, 7, 0, 7, 1, 1, 7, 5],
        [9, 0, 3, 9, 3, 5, 5, 3, 7],
        [9, 8, 7, 5, 9, 7],
        [5, 8, 4, 5, 10, 8, 10, 11, 8],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
        [9, 4, 5, 2, 11, 3],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
        [5, 10, 2, 5, 2, 4, 4, 2, 0],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
        [8, 4, 5, 8, 5, 3, 3, 5, 1],
        [0, 4, 5, 1, 0, 5],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
        [9, 4, 5],
        [4, 11, 7, 4, 9, 11, 9, 10, 11],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
        [11, 7, 4, 11, 4, 2, 2, 4, 0],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
        [1, 10, 2, 8, 7, 4],
        [4, 9, 1, 4, 1, 7, 7, 1, 3],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
        [4, 0, 3, 7, 4, 3],
        [4, 8, 7],
        [9, 10, 8, 10, 11, 8],
        [3, 0, 9, 3, 9, 11, 11, 9, 10],
        [0, 1, 10, 0, 10, 8, 8, 10, 11],
        [3, 1, 10, 11, 3, 10],
        [1, 2, 11, 1, 11, 9, 9, 11, 8],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
        [0, 2, 11, 8, 0, 11],
        [3, 2, 11],
        [2, 3, 8, 2, 8, 10, 10, 8, 9],
        [9, 10, 2, 0, 9, 2],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
        [1, 10, 2],
        [1, 3, 8, 9, 1, 8],
        [0, 9, 1],
        [0, 3, 8],
        []
    ]
    edgeShifts = np.array([
        # #maps edge ID (0-11) to (x,y,z) cell offset and edge ID (0-2)
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
        [1, 0, 0, 2],
        [1, 1, 0, 2],
        [0, 1, 0, 2],
        # [9, 9, 9, 9]  ## fake
    ], dtype=np.uint16)
    # don't use ubyte here! This value gets added to cell index later;
    # will need the extra precision.
    nTableFaces = np.array(
        [len(f)/3 for f in triTable], dtype=np.ubyte)
    faceShiftTables = [None]
    for i in range(1, 6):
        # # compute lookup table of index: vertexes mapping
        faceTableI = np.zeros((len(triTable), i*3), dtype=np.ubyte)
        faceTableInds = np.argwhere(nTableFaces == i)

        faceTableI[faceTableInds[:, 0]] = np.array(
                [triTable[j] for j in faceTableInds[:, 0]])
        faceTableI = faceTableI.reshape((len(triTable), i, 3))
        faceShiftTables.append(edgeShifts[faceTableI])

    return faceShiftTables, edgeShifts, edgeTable, nTableFaces


IsosurfaceDataCache = None


def isosurface(data, level):
    """
    Generate isosurface from volumetric data using marching cubes algorithm.
    See Paul Bourke, "Polygonising a Scalar Field"
    (http://paulbourke.net/geometry/polygonise/)

    *data*   3D tensor of scalar values.
    *level*  The level at which to generate an isosurface

    Returns an array of vertex coordinates (Nv, 3) and an array of
    per-face vertex indexes (Nf, 3)
    """
    # For improvement, see:
    ##
    # Efficient implementation of Marching Cubes' cases with topological
    # guarantees.
    # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
    # Journal of Graphics Tools 8(2): pp. 1-15 (december 2003)

    # Precompute lookup tables on the first run
    global IsosurfaceDataCache
    if IsosurfaceDataCache is None:
        IsosurfaceDataCache = _get_isosurface_data()
    faceShiftTables, edgeShifts, edgeTable, nTableFaces = \
        IsosurfaceDataCache

    faceShiftTables_tf = tuple(
        (None if f is None else tf.constant(f, tf.int32)
         for f in faceShiftTables))
    nTableFaces_tf = tf.constant(nTableFaces, dtype=tf.int32)

    # mark everything below the isosurface level
    mask = tf.cast(data < level, tf.int32)

    # make eight sub-fields and compute indexes for grid cells
    updates = []
    slices = [slice(0, -1), slice(1, None)]
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                # this is just to match Bourk's vertex numbering scheme
                vertIndex = i - 2 * j * i + 3 * j + 4 * k
                m = mask[slices[i], slices[j], slices[k]]
                updates.append(m * 2 ** vertIndex)
    index = tf.add_n(updates)

    # Generate table of edges that have been cut
    cutEdges = [[], [], []]

    edgeTable = tf.constant(edgeTable, dtype=tf.uint16)
    edges = tf.gather(edgeTable, index)
    for i, shift in enumerate(edgeShifts):
        update = tf.bitwise.bitwise_and(edges, 2**i)
        update = tf.pad(update, [[s, 1 - s] for s in shift[:3]])
        cutEdges[shift[3]].append(update)
    cutEdges = [tf.cast(tf.add_n(c), tf.int32) for c in cutEdges]
    cutEdges = tf.stack(cutEdges, axis=-1)

    # for each cut edge, interpolate to see where exactly the edge is cut and
    # generate vertex positions
    m = cutEdges > 0
    vertexInds = tf.where(m)
    vertexes = tf.cast(vertexInds[:, :3], tf.float32)

    # re-use the cutEdges array as a lookup table for vertex IDs
    update = tf.range(tf.shape(vertexInds)[0])
    cutEdges = gather_updated(cutEdges, vertexInds, update)

    assert(index.dtype == tf.int32)
    # index = tf.cast(index, tf.int32)

    vs = tf.unstack(vertexInds, axis=-1)
    vertexes_unstacked = tf.unstack(vertexes, axis=1)
    for i in [0, 1, 2]:
        vim = tf.equal(vs[3], i)
        vi1 = tf.boolean_mask(vertexInds, vim)
        vss = tf.unstack(vi1, axis=1)[:3]
        vi1 = tf.stack(vss, axis=1)
        vss[i] += 1
        vi2 = tf.stack(vss, axis=1)
        v1 = tf.gather_nd(data, vi1)
        v2 = tf.gather_nd(data, vi2)

        update = (level - v1) / (v2 - v1)
        vertexes_unstacked[i] = scatter_added(
            vertexes_unstacked[i], vim, update)

    vertexes = tf.stack(vertexes_unstacked, axis=1)

    # compute the set of vertex indexes for each face.

    # This works, but runs a bit slower.
    # # all cells with at least one face
    # cells = np.argwhere((index != 0) & (index != 255))
    # cellInds = index[cells[:,0], cells[:,1], cells[:,2]]
    # verts = faceTable[cellInds]
    # mask = verts[...,0,0] != 9
    # # we now have indexes into cutEdges
    # verts[...,:3] += cells[:,np.newaxis,np.newaxis,:]
    # verts = verts[mask]
    # faces = cutEdges[verts[...,0], verts[...,1], verts[...,2], verts[...,3]]
    # ## and these are the vertex indexes we want.

    # To allow this to be vectorized efficiently, we count the number of faces
    # in each grid cell and handle each group of cells with the same number
    # together.
    # determine how many faces to assign to each grid cell
    nFaces = tf.gather(nTableFaces_tf, index)

    faces = []

    # cutEdges = tf.constant(cutEdges, dtype=tf.int32)
    if not isinstance(index, tf.Tensor):
        index = tf.constant(index, dtype=tf.int32)
    elif index.dtype != tf.int32:
        index = tf.cast(index, tf.int32)

    for i in range(1, 6):
        # expensive:
        # all cells which require i faces  (argwhere is expensive)
        cells = tf.where(tf.equal(nFaces, i))
        cells = tf.cast(cells, tf.int32)
        # index values of cells to process for this round
        cellInds = tf.gather_nd(index, cells)

        # expensive:
        verts = tf.gather(faceShiftTables_tf[i], cellInds)
        v0, v1 = tf.split(verts, [3, 1], axis=-1)
        s0, s1 = (-1 if s is None else s for s in cells.shape.as_list())
        cells = tf.reshape(cells, (s0, 1, 1, s1))
        v0 = v0 + cells
        verts = tf.concat([v0, v1], axis=-1)

        verts = tf.reshape(verts, [-1] + verts.shape.as_list()[2:])

        # expensive:
        vertInds = tf.gather_nd(cutEdges, verts)
        faces.append(vertInds)
    faces = tf.concat(faces, axis=0)

    return vertexes, faces

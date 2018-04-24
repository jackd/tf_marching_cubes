# Tensorflow Marching Cubes Implementation
Basic implementation of marching cubes in tensorflow for extracting isosurfaces from embedding functions.

Based on the numpy implementation from pyqt. See `np_impl.py` for reference implementation.

Note the implementation uses only native tensorflow operations (no `tf.py_func`), so is piece-wise differentiable. See [`example/learn.py`](https://github.com/jackd/tf_marching_cubes/blob/master/example/learn.py).

## Example usage
```
import tensorflow as tf
import numpy as np
from tf_marching_cubes import isosurface

data = np.load('data_path.npy')
level = 0
print(data.shape)  # (p, q, r)
verts, faces = isosurface(tf.constant(data, dtype=tf.float32), level)

with tf.Session() as sess:
  vertex_data, face_data = sess.run([verts, faces])
```

See [`example`](https://github.com/jackd/tf_marching_cubes/tree/master/example) directory for more details including use in batches.

## Setup
Clone, and add the parent directory to your python path
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_marching_cubes.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

## Requirements
The code has no external dependencies other than `numpy` (developed with v1.14.2) and `tensorflow` (developed with v1.6).

### Examples
Examples require `mayavi` for visualization, `simple.py` requires `skimage` and `learn.py` benefits from `progress`. Developed/tested on `python` 2.7.12 but should be easily ported to python 3.

```
pip install mayavi scikit-image progress
```

## Notes
* Slow. No  serious attempt at optimizing has been made, and the current implementation makes extensive use of `tf.gather` and `tf.gather_nd`, both notoriously slow operations.
* Variable-sized output. See `example/batch.py` for an example batch usage.
* Differentiable. See `example/learn.py` for evidence.

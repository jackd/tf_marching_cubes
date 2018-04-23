# Tensorflow Marching Cubes Implementation
Basic implementation of marching cubes in tensorflow for extracting isosurfaces from embedding functions.

Based on the numpy implementation from pyqt. See `np_impl.py` for reference implementation.

Note the tensorflow implementation does not use `tf.py_func`, so should be piece-wise differentiable.

## Setup
Clone, and add the parent directory to your python path
```
cd /path/to/parent_dir
git clone
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

## Notes
* Slow. No attempt at optimizing has been made, and the current implementation makes extensive use of `tf.gather` and `tf.gather_nd`, both notoriously slow operations.
* Variable-sized output. For use in batches, `isosurface` will need to be combined with another function to map the vertices and faces to some fixed sized output and used with `tf.map`. See `example/batch.py`.
* Differentiable. See `example/learn.py` for evidence.

## TODO
* `IsosurfaceDataCache` is based on numpy implementation. Replace by checking named tensors exist in graph.

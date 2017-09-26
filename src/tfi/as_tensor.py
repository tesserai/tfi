import tensorflow as tf
import collections
import numpy as np

def as_tensor(object, shape, dtype):
    if isinstance(object, (str, np.ndarray)):
        return object
    if isinstance(object, collections.Iterable):
        if shape is None:
            max_len = None
        elif len(shape) >= 1:
            max_len = shape[0]
        else:
            raise Exception("Shape is %s. Didn't expect iterable: %s" % (shape, object))
        sub_shape = shape[1:] if shape is not None else None
        r = []
        cur_len = 0
        for o in object:
            r.append(as_tensor(o, sub_shape, dtype))
            cur_len += 1
            if max_len is not None and cur_len > max_len:
                raise Exception("Iterable too long. Expected exactly %s elements from: %s" % (max_len, object))
        return tf.stack(r)
    if hasattr(object, '__tensor__'):
        return object.__tensor__(shape, dtype)
    return object

from functools import reduce

def _shape_match_score(src_shape, xfrm_shape):
    # Returns (0|1|2, <n exact matching dimensions>)

    compatible = True
    if src_shape is None or xfrm_shape is None:
        matching_dims = 0
        exact = src_shape == xfrm_shape
    else:
        matching_dims = 0
        # Can only be exact if lengths match.
        exact = len(src_shape) == len(xfrm_shape)
        # TODO(adamb) Prove this fails and then loop over dims in reverse.
        for a, b in zip(reversed(src_shape), reversed(xfrm_shape)):
            if a == b:
                matching_dims += 1
            elif a is not None and b is not None:
                compatible = False
                exact = False
            else:
                exact = False

    if exact:
        return (2, matching_dims)

    if src_shape == None or xfrm_shape == None:
        return (1, matching_dims)

    # If src_shape has at least as many dimensions as xfrm_shape expects,
    # it's a partial match.
    if len(src_shape) >= len(xfrm_shape) and compatible:
        return (1, matching_dims)

    return (0, matching_dims)

def from_tensor(tensor, mappings):
    if tensor is None:
        return tensor

    if not isinstance(tensor, (tf.Tensor, np.ndarray)):
        return tensor

    if isinstance(tensor, (tf.Tensor)):
        with tf.Session():
            tensor = tensor.eval()

    shape_dims = tensor.shape
    if isinstance(shape_dims, tf.TensorShape):
        shape_dims = shape_dims.dims
    dtype = tensor.dtype

    matches = [
        (_shape_match_score(shape_dims, needs_shape_dims), needs_shape_dims, map_fn)
        for needs_dtype, needs_shape_dims, map_fn in mappings
        if needs_dtype == dtype
    ]

    if len(matches) == 0:
        return None

    matches.sort(reverse=True)
    for (score, nmatching), needs_shape_dims, fn in matches:
        if score == 0:
            break

        if score == 2:
            return fn(tensor)

        if score == 1:
            needs_ndims = len(needs_shape_dims)
            matching_dims = shape_dims[-needs_ndims:]
            remaining_dims = shape_dims[:-needs_ndims]
            if len(remaining_dims) == 0:
                return fn(tensor)

            return np.reshape([
                fn(a)
                for a in np.reshape(tensor, [-1, *matching_dims])
            ], remaining_dims)

    return None

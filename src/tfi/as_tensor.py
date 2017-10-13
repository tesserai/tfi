import tensorflow as tf
import collections
import numpy as np

class ShapeMismatchException(Exception):
    def __init__(self, dtype, longest_matching_suffix):
        self.dtype = dtype
        self.longest_matching_suffix = longest_matching_suffix

def _detect_native_kind(object):
    if isinstance(object, str):
        return [], tf.string, lambda o, shp: np.reshape(o, shp)
    if isinstance(object, int):
        return [], tf.int32, lambda o, shp: np.reshape(o, shp)
    if isinstance(object, float):
        return [], tf.float32, lambda o, shp: np.reshape(o, shp)
    return None, None, None

def _iterable_as_tensor(object, shape, dtype):
    # TODO(adamb) For now assume that elements in iterable can be stacked.
    #     Should actually check that they all have the same shape and dtype.
    #     If they don't, we should be returning a list of them, not a tensor.

    # If object is iterable, assume we'll make a list of tensors with
    # adjusted target shapes.
    if shape is None:
        expect_len = None
    elif len(shape) >= 1:
        # Perhaps we know enough about the target shape to know expected length.
        expect_len = shape[0]
    else:
        raise Exception("Shape is %s. Didn't expect iterable: %s" % (shape, object))
    r = []
    cur_len = 0
    # Adjust target shape for recursive call.
    sub_shape = shape[1:] if shape is not None else None
    for o in object:
        r.append(as_tensor(o, sub_shape, dtype))
        cur_len += 1
        if expect_len is not None and cur_len > expect_len:
            break
    if expect_len is not None and cur_len != expect_len:
        raise Exception("Expected exactly %s elements (got at least %s) from: %s" % (expect_len, cur_len, object))
    return tf.stack(r)

def as_tensor(object, shape, dtype):
    # TODO(adamb) Figure out how to handle coercing primitive types to TF dtypes.
    candidate_dims, candidate_dtype, reshape_fn = _detect_native_kind(object)

    if reshape_fn is not None:
        candidate = object
    elif isinstance(object, np.ndarray):
        candidate = tf.constant(object)
        candidate_dims = object.shape
        candidate_dtype = tf.as_dtype(object.dtype)
        reshape_fn = lambda o, shp: np.reshape(o, shp)
    elif isinstance(object, collections.Iterable):
        candidate = _iterable_as_tensor(object, shape, dtype)
        candidate_dims = candidate.shape.dims
        candidate_dtype = candidate.dtype
        reshape_fn = lambda o, shp: tf.reshape(o, shp)
    elif hasattr(object, '__tensor__'):
        # Assume that we need to use __tensor__ method if object doesn't have
        # a native dtype.
        try:
            candidate = object.__tensor__(shape, dtype)
            candidate_dims = candidate.shape.dims
            candidate_dtype = candidate.dtype
            reshape_fn = lambda o, shp: tf.reshape(o, shp)
        except ShapeMismatchException as sme:
            candidate = object
            candidate_dims = sme.longest_matching_suffix
            candidate_dtype = sme.dtype
            reshape_fn = lambda o, shp: tf.reshape(o.__tensor__(candidate_dims, dtype), shp)
    else:
        raise Exception("Could not coerce %s to tensor %s with shape %s" % (object, dtype, shape))

    if tf.TensorShape(candidate_dims).is_compatible_with(tf.TensorShape(shape)):
        return candidate

    candidate_ndims = len(candidate_dims)
    if candidate_ndims > shape.ndims or shape.dims[-candidate_ndims:] != candidate_dims:
        # Returned shape must be an exact suffix of target shape.
        raise Exception("Could not coerce %s to tensor %s with shape %s. Candidate dimensions are not a compatible suffix: %s" % (object, dtype, shape, candidate_dims))
    for dim in shape.dims[:-candidate_ndims]:
        # Check that all remaining dimensions are either None or 1.
        if dim is not None and dim != 1:
            raise Exception("Could not coerce %s to tensor %s with shape %s. There are dimensions in target shape that prevent simple reshaping of compromise shape: %s" % (object, dtype, shape, compromise))
    return reshape_fn(candidate, shape)

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

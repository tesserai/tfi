import collections
import numpy as np

from functools import reduce

# TODO(adamb) Rename shape to dims everywhere that it's just a list/tuple.

class ShapeMismatchException(Exception):
    def __init__(self, dtype, longest_matching_suffix):
        self.dtype = dtype
        self.longest_matching_suffix = longest_matching_suffix

class _BaseAdapter(object):

    def _iterable_as_tensor(self, object, shape, dtype):
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
        candidate_dtype = dtype
        check_dtype = lambda o: isinstance(o, candidate_dtype)
        for o in object:

            r.append(self.as_tensor(o, sub_shape, dtype))
            cur_len += 1
            if expect_len is not None and cur_len > expect_len:
                break
        if expect_len is not None and cur_len != expect_len:
            raise Exception("Expected exactly %s elements (got at least %s) from: %s" % (expect_len, cur_len, object))
        return self._stack(r)

    def as_tensor(self, object, shape, dtype):
        tensor = self.maybe_as_tensor(object, shape, dtype)
        if tensor is None:
            raise Exception("Could not coerce %s %s to tensor %s with shape %s" % (object, type(object), dtype, shape))
        return tensor

    def maybe_as_tensor(self, object, shape, dtype):
        candidate, candidate_dims, candidate_dtype, reshape_fn = self._detect_native_kind(object)

        if reshape_fn is not None:
            pass
        elif not isinstance(object, (str, dict)) and isinstance(object, collections.Iterable):
            candidate = self._iterable_as_tensor(object, shape, dtype)
            candidate_dims = self._tensor_dims(candidate)
            candidate_dtype = self._tensor_dtype(candidate)
            reshape_fn = self._reshape_tensor
        elif hasattr(object, '__tensor__'):
            # Assume that we need to use __tensor__ method if object doesn't have
            # a native dtype.
            try:
                candidate = object.__tensor__(shape, dtype)
                candidate_dims = self._tensor_dims(candidate)
                candidate_dtype = self._tensor_dtype(candidate)
                reshape_fn = self._reshape_tensor
            except ShapeMismatchException as sme:
                candidate = object
                candidate_dims = sme.longest_matching_suffix
                candidate_dtype = sme.dtype
                reshape_fn = lambda o, shp: self._reshape_tensor(
                        o.__tensor__(candidate_dims, dtype),
                        [-1 if d is None else d for d in shp])
        else:
            return None

        if self._are_shapes_compatible(candidate_dims, shape):
            return candidate

        candidate_ndims = len(candidate_dims)
        if candidate_ndims > len(shape) or shape[-candidate_ndims:] != candidate_dims:
            # Returned shape must be an exact suffix of target shape.
            raise Exception("Could not coerce %s %s to tensor %s with shape %s. Candidate dimensions are not a compatible suffix: %s" % (object, type(object), dtype, shape, candidate_dims))
        for dim in shape[:-candidate_ndims]:
            # Check that all remaining dimensions are either None or 1.
            if dim is not None and dim != 1:
                raise Exception("Could not coerce %s %s to tensor %s with shape %s. There are dimensions in target shape that prevent simple reshaping of compromise shape: %s" % (object, type(object), dtype, shape, candidate_dims))
        return reshape_fn(candidate, shape)

    def _shape_match_score(self, src_shape, xfrm_shape):
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

    def from_tensor(self, tensor, mappings):
        if tensor is None:
            return tensor

        tensor = self._as_ndarray(tensor)

        if not isinstance(tensor, np.ndarray):
            return tensor

        shape_dims = tensor.shape_dims
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

import torch
class PyTorchAdapter(_BaseAdapter):
    def _stack(self, r):
        return torch.stack(r)

    def _reshape_tensor(self, o, shp):
        return o.view(shp)

    def _tensor_dims(self, tensor):
        return tensor.size()

    def _tensor_dtype(self, tensor):
        return type(tensor)

    def _are_shapes_compatible(self, candidate_dims, shape):
        if candidate_dims == None or shape == None:
            return True
        if len(candidate_dims) != len(shape):
            return False
        for d1, d2 in zip(candidate_dims, shape):
            if d1 != d2 and d1 != None and d2 != None:
                return False
        return True

    def _detect_native_kind(self, object):
        if isinstance(object, int):
            return object, [], np.int32, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, float):
            return object, [], np.float32, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, np.ndarray):
            return (
                object,
                object.shape,
                object.dtype,
                lambda o, shp: np.reshape(o, shp)
            )
        return object, None, None, None

    # def _maybe_as_ndarray(self, tensor):
    #     return tensor

# import tensorflow as tf
class TensorFlowAdapter(_BaseAdapter):
    def _stack(self, r):
        return tf.stack(r)

    def _tensor_dims(self, tensor):
        return tensor.shape.dims

    def _tensor_dtype(self, tensor):
        return tensor.dtype

    def _reshape_tensor(o, shp):
        return tf.reshape(o, shp)

    def _are_shapes_compatible(self, candidate_dims, shape):
        return tf.TensorShape(candidate_dims).is_compatible_with(tf.TensorShape(shape))

    def _detect_native_kind(self, object):
        if isinstance(object, str):
            return object, [], tf.string, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, int):
            return object, [], tf.int32, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, float):
            return object, [], tf.float32, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, np.ndarray):
            return (
                tf.constant(object),
                object.shape,
                tf.as_dtype(object.dtype),
                lambda o, shp: np.reshape(o, shp)
            )
        return object, None, None, None

    # def _maybe_as_ndarray(self, tensor):
    #     if isinstance(tensor, tf.Tensor):
    #         with tf.Session():
    #             return tensor.eval()
    #     return tensor



# _x = TensorFlowAdapter()
_x = PyTorchAdapter()
as_tensor = _x.as_tensor
maybe_as_tensor = _x.maybe_as_tensor
from_tensor = _x.from_tensor

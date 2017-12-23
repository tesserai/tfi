import collections
import numpy as np

from functools import reduce

# TODO(adamb) Rename shape to dims everywhere that it's just a list/tuple.

class ShapeMismatchException(Exception):
    def __init__(self, dtype, longest_matching_suffix):
        self.dtype = dtype
        self.longest_matching_suffix = longest_matching_suffix

class _BaseAdapter(object):
    def __init__(self, adapter):
        self._adapter = adapter

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
        return self._adapter._stack(r)

    def decode_bytes(self, *args):
        return self._adapter._decode_bytes(*args)

    def decode_open_file(self, *args):
        return self._adapter._decode_open_file(*args)

    def decode_file_path(self, *args):
        return self._adapter._decode_file_path(*args)

    def as_tensor(self, object, shape, dtype):
        tensor = self.maybe_as_tensor(object, shape, dtype)
        if tensor is None:
            raise Exception("Could not coerce %s %s to tensor %s with shape %s" % (object, type(object), dtype, shape))
        return tensor

    def maybe_as_tensor(self, object, shape, dtype):
        candidate, candidate_dims, candidate_dtype, reshape_fn = self._adapter._detect_native_kind(object)

        if reshape_fn is not None:
            pass
        elif not isinstance(object, (str, dict)) and isinstance(object, collections.Iterable):
            candidate = self._iterable_as_tensor(object, shape, dtype)
            candidate_dims = self._adapter._tensor_dims(candidate)
            candidate_dtype = self._adapter._tensor_dtype(candidate)
            reshape_fn = self._adapter._reshape_tensor
        elif hasattr(object, '__tensor__'):
            # Assume that we need to use __tensor__ method if object doesn't have
            # a native dtype.
            try:
                candidate = object.__tensor__(self, shape, dtype)
                candidate_dims = self._adapter._tensor_dims(candidate)
                candidate_dtype = self._adapter._tensor_dtype(candidate)
                reshape_fn = self._adapter._reshape_tensor
            except ShapeMismatchException as sme:
                candidate = object
                candidate_dims = sme.longest_matching_suffix
                candidate_dtype = sme.dtype
                reshape_fn = lambda o, shp: self._adapter._reshape_tensor(
                        o.__tensor__(self, candidate_dims, dtype),
                        [-1 if d is None else d for d in shp])
        else:
            return None

        if self._adapter._are_shapes_compatible(candidate_dims, shape):
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

def _from_tensor(tensor, shape_dims, dtype, reshape_fn, mappings):
    matches = [
        (_shape_match_score(shape_dims, needs_shape_dims), needs_shape_dims, map_fn)
        for needs_dtype, needs_shape_dims, map_fn in mappings
        if needs_dtype == dtype
    ]

    if len(matches) == 0:
        return None

    # Descending order means we'll consider highest score first.
    matches.sort(reverse=True)
    (score, nmatching), needs_shape_dims, xform = matches[0]
    if score == 0:
        return None

    # Perfect match.
    if score == 2:
        # Transform the entire tensor directly.
        return xform(tensor)

    # Partial match, but still best possible.
    needs_ndims = len(needs_shape_dims)
    matching_dims = shape_dims[-needs_ndims:]
    remaining_dims = shape_dims[:-needs_ndims]
    if len(remaining_dims) == 0:
        return xform(tensor)

    return reshape_fn([
        xform(a)
        for a in reshape_fn(tensor, [-1, *matching_dims])
    ], remaining_dims)


_ENCODERS = []
def register_encoder(_mimetypes, _dtypes, _shapes):
    def _register(func):
        for dtype in _dtypes:
            for shape in _shapes:
                for mimetype in _mimetypes:
                    _ENCODERS.append((mimetype, dtype, shape, func))
        return func
    return _register

_REFLECTORS = []
def register_tensor_spec(cls):
    def _register(func):
        _REFLECTORS.append((cls, func))
        return func
    return _register

def _tensor_spec(tensor):
    for cls, func in _REFLECTORS:
        if isinstance(tensor, cls):
            return func(tensor)
    return None, None, None, None

def _compose(f, g):
    return lambda *a: f(g(*a))

def encode(accept_mimetypes, tensor):
    if tensor is None:
        return tensor

    candidate, shape_dims, dtype, reshape_fn = _tensor_spec(tensor)
    if shape_dims is None:
        return tensor

    result = _from_tensor(
            candidate,
            shape_dims,
            dtype,
            reshape_fn,
            [
                (dtype, shape, _compose(accept_mimetypes[mimetype], encoder))
                for mimetype, dtype, shape, encoder in _ENCODERS
                if mimetype in accept_mimetypes
            ])

    return tensor if result is None else result

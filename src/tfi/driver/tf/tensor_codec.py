import tensorflow as tf
import numpy as np

import mimetypes

from tfi.tensor.codec import ShapeMismatchException as _ShapeMismatchException

from tfi.base import _recursive_transform
import tensorflow as tf

_DECODERS = {}
def _register_decoder(_mimetypes):
    def _register(func):
        for mimetype in _mimetypes:
            _DECODERS[mimetype] = func
        return func
    return _register

def _decode(mimetype, *args):
    func = _DECODERS[mimetype]
    return func(*args)

import json
@_register_decoder(["text/json"])
def _decode_json(shape, dtype, bytes):
    r = json.loads(bytes)
    return tf.constant(r, shape=shape, dtype=dtype)

@_register_decoder(["image/jpeg", "image/png"])
def _decode_image(shape, dtype, bytes):
    channels = None
    h = None
    w = None
    if shape is not None:
        shape_len = len(shape)
        # NOTE(adamb) Technically we can load images, convert them to
        #     single channel (if they aren't already) and reshape them to
        #     [h, w]. Revisit the code below if we ever want this.
        if shape_len < 3:
            return None
        # NOTE(adamb) Technically we can handle .gif files, which decode to
        #     a shape of length 4. For this to work properly, we'll need to
        #     know which one we're dealing with and return None if the
        #     format doesn't match the requested shape.
        if shape_len > 3:
            raise _ShapeMismatchException(dtype, shape[-3:])
        channels = shape[-1]
        h = shape[-3]
        w = shape[-2]

    # contents = _as_tensor(bytes, None, None)
    image = tf.image.decode_image(bytes, channels=channels)

    # Assumes we're decoding a jpeg or png, not gif.
    image.set_shape([None, None, None])

    if dtype is not None and dtype != image.dtype:
        image = tf.image.convert_image_dtype(image, dtype=dtype)

    if w is not None or h is not None:
        image = tf.image.resize_images(image, tf.constant([h, w]))

    return image


class _driver(object):
    def _decode_file_path(self, shape, dtype, mimetype, path):
        return _decode(
                mimetype,
                shape,
                dtype,
                tf.read_file(path))

    def _decode_open_file(self, shape, dtype, mimetype, handle):
        pos = handle.tell()
        try:
            bytes = handle.read()
            return _decode(mimetype, shape, dtype, bytes)
        finally:
            handle.seek(pos)

    def _decode_bytes(self, shape, dtype, mimetype, bytes):
        return _decode(mimetype, shape, dtype, bytes)

    def _stack(self, r):
        return tf.stack(r)

    def _tensor_dims(self, tensor):
        return tensor.shape.dims

    def _tensor_dtype(self, tensor):
        return tensor.dtype

    def _reshape_tensor(self, o, shp):
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
        if isinstance(object, list) and len(object) > 0:
            _, shape, dtype, reshape = self._detect_native_kind(object[0])
            return object, [len(object), *shape], dtype, lambda o, shp: np.reshape(o, shp)
        if isinstance(object, np.ndarray):
            return (
                tf.constant(object),
                object.shape,
                tf.as_dtype(object.dtype),
                lambda o, shp: np.reshape(o, shp)
            )
        return object, None, None, None

from tfi.tensor.codec import _BaseAdapter

_as_tensor_adapter = _BaseAdapter(_driver())
as_tensor = _as_tensor_adapter.as_tensor


import tensorflow as tf
import numpy as np

from tfi.data import _Source

from tfi.tensor.codec import register_encoder as _register_encoder
from tfi.tensor.codec import register_tensor_spec as _register_tensor_spec


@_register_tensor_spec(_Source)
def _file_source_tensor_spec(fp):
    tensor = as_tensor(fp, None, None)
    return tensor, tensor.shape, tensor.dtype, np.reshape

@_register_tensor_spec(np.ndarray)
def _ndarray_spec(tensor):
    return tensor, tensor.shape, tensor.dtype, np.reshape

def _graph_for_tensor(tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.graph
    return tf.Graph()

@_register_encoder(
        ["image/png"],
        [tf.uint8],
        [(None, None, None), (None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
def _png_encode(tensor):
    with _graph_for_tensor(tensor).as_default() as g:
        with tf.Session(graph=g):
            # TODO(adamb) Use placeholder and a cached graph, for speed.
            encoded = tf.image.encode_png(tensor)
            return encoded.eval()

@_register_encoder(
        ["image/png"],
        [tf.float32],
        [(None, None, None), (None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
def _png_float_encode(tensor):
    with _graph_for_tensor(tensor).as_default() as g:
        with tf.Session(graph=g):
            # TODO(adamb) Use placeholder and a cached graph, for speed.
            encoded = tf.image.encode_png(tf.cast(tensor * 255.0, tf.uint8))
            return encoded.eval()

@_register_encoder(
        ["python/jsonable"],
        [np.float32],
        [None])
def _jsonable_encode_ndarray(tensor):
    return tensor.tolist()

@_register_encoder(
        ["python/jsonable"],
        [object],
        [(None)])
def _jsonable_encode(tensor):
    return [o.decode() if isinstance(o, bytes) else o for o in tensor]

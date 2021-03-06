import json
from google.protobuf.json_format import ParseDict

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

def _decode(mimetype, shape, dtype, bytes):
    if not mimetype:
        return bytes
    func = _DECODERS[mimetype]
    return func(shape, dtype, bytes)

import json
@_register_decoder(["text/json"])
def _decode_json(shape, dtype, bytes):
    r = json.loads(bytes)
    return tf.constant(r, shape=shape, dtype=dtype)

@_register_decoder(["image/jpeg", "image/png"])
def _decode_image(shape, dtype, bytes):
    if dtype == tf.string:
        if shape is None or len(shape) == 0:
            return bytes

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
        print("will convert image dtype from", image.dtype, "to", dtype)
        image = tf.image.convert_image_dtype(image, dtype=dtype)

    if w is not None or h is not None:
        image = tf.image.resize_images(image, tf.constant([h, w]))

    return image


def _np_reshape_tensor(o, shp):
    shp = [dim if dim is not None else -1 for dim in shp]
    return np.reshape(o, shp)


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

    def _tensor_candidate(self, obj, shape, dtype):
        if dtype == tf.string and (shape is None or len(shape) == 0):
            if isinstance(obj, dict) and 'features' in obj:
                return (
                    ParseDict(obj, tf.train.Example()).SerializeToString(),
                    [],
                    tf.string,
                    _np_reshape_tensor,
                )
                    

        return None, None, None, None

    def _reshape_tensor(self, o, shp):
        return tf.reshape(o, shp)

    def _are_shapes_compatible(self, candidate_dims, shape):
        return tf.TensorShape(candidate_dims).is_compatible_with(tf.TensorShape(shape))

    def _detect_native_kind(self, obj):
        if isinstance(obj, (str, bytes)):
            return obj, [], tf.string, _np_reshape_tensor
        if isinstance(obj, int):
            return obj, [], tf.int32, _np_reshape_tensor
        if isinstance(obj, float):
            return obj, [], tf.float32, _np_reshape_tensor
        if isinstance(obj, np.ndarray):
            return (
                tf.constant(obj),
                obj.shape,
                tf.as_dtype(obj.dtype),
                _np_reshape_tensor
            )
        if isinstance(obj, list) and len(obj) > 0:
            _, shape, dtype, reshape = self._detect_native_kind(obj[0])
            # Perhaps this is a "native" list? Recurse if so.
            if reshape is not None:
                nd = np.array(obj, ndmin=len(shape)+1)
                return nd, [len(obj), *shape], dtype, _np_reshape_tensor
        return obj, None, None, None

from tfi.tensor.codec import _BaseAdapter

_as_tensor_adapter = _BaseAdapter(_driver())
as_tensor = _as_tensor_adapter.as_tensor


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
        ["image/jpeg"],
        [tf.float32],
        [(None, None, None), (None, None, 1), (None, None, 2), (None, None, 3)])
def _jpeg_float_encode(tensor):
    with _graph_for_tensor(tensor).as_default() as g:
        with tf.Session(graph=g):
            # TODO(adamb) Use placeholder and a cached graph, for speed.
            encoded = tf.image.encode_jpeg(tf.cast(tensor * 255.0, tf.uint8))
            return encoded.eval()

@_register_encoder(
        ["python/jsonable"],
        [np.float32, np.ndarray],
        [None])
def _jsonable_encode_ndarray(tensor):
    return tensor.tolist()

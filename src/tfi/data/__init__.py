import mimetypes

from tfi.as_tensor import as_tensor, from_tensor

import tensorflow as tf

_DECODERS = {}
def _register_decoder(_mimetypes):
    def _register(func):
        for mimetype in _mimetypes:
            _DECODERS[mimetype] = func
        return func
    return _register

def _decode(mimetype, data):
    func = _DECODERS[mimetype]
    return func(data)

class Decoder:
    pass

@_register_decoder(["image/jpeg", "image/png"])
class _ImageBytes(Decoder):
    def __init__(self, bytes):
        self.bytes = bytes

    def __tensor__(self, shape, dtype):
        channels = None
        h = None
        w = None
        if shape is not None:
            shape_len = len(shape)
            if shape_len < 3 or shape_len > 4:
                raise Exception("Unsupported shape: %s" % shape)
            channels = shape[-1]
            h = shape[-3]
            w = shape[-2]

        contents = as_tensor(self.bytes, None, None)
        image = tf.image.decode_image(contents, channels=channels)
        if dtype is not None and dtype != image.dtype:
            image = tf.image.convert_image_dtype(image, dtype=dtype)

        if w is not None or h is not None:
            image = tf.image.resize_image_with_crop_or_pad(image, target_height=h, target_width=w)

        return image

_ENCODERS = []
def _register_encoder(_mimetypes, _dtypes, _shapes):
    def _register(func):
        for dtype in _dtypes:
            for shape in _shapes:
                for mimetype in _mimetypes:
                    _ENCODERS.append((mimetype, dtype, shape, func))
        return func
    return _register

def _compose(f, g):
    return lambda x: f(g(x))

def _encode(tensor, accept_mimetypes):
    return from_tensor(
            tensor,
            [
                (dtype, shape, _compose(accept_mimetypes[mimetype], encoder))
                for mimetype, dtype, shape, encoder in _ENCODERS
                if mimetype in accept_mimetypes
            ])

@_register_encoder(
        ["image/png"],
        [tf.uint8],
        [(None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
def _png_encode(tensor):
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g):
            # TODO(adamb) Use placeholder and a cached graph, for speed.
            encoded = tf.image.encode_png(tensor)
            return encoded.eval()

@_register_encoder(
        ["image/png"],
        [tf.float32],
        [(None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
def _png_float_encode(tensor):
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g):
            # TODO(adamb) Use placeholder and a cached graph, for speed.
            encoded = tf.image.encode_png(tf.cast(tensor * 255.0, tf.uint8))
            return encoded.eval()

class _FileBytes(Decoder):
    def __init__(self, path):
        self._path = path

    def __tensor__(self, shape, dtype):
        return tf.read_file(self._path)

def file(path):
    mimetype, _ = mimetypes.guess_type("file://%s" % path)
    return _decode(mimetype, _FileBytes(path))

import pprint
import sys
from tfi.data.terminal import imgcat

def terminal_write(o):
    tensor = as_tensor(o, None, None)
    accept_mimetypes = {"image/png": imgcat, "text/plain": lambda x: x}
    encoded = _encode(tensor, accept_mimetypes)
    if encoded is None:
        encoded = tensor
    print(pprint.pformat(encoded), file=sys.stdout, flush=True)

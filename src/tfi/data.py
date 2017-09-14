import mimetypes

from tfi.as_tensor import as_tensor

import tensorflow as tf

_DECODERS = {}
def _register_mimetypes(_mimetypes):
    def _register(func):
        for mimetype in _mimetypes:
            _DECODERS[mimetype] = func
        return func
    return _register

def _decode(mimetype, data):
    func = _DECODERS[mimetype]
    return func(data)

@_register_mimetypes(["image/jpeg", "image/png"])
class _ImageBytes:
    def __init__(self, bytes):
        self.bytes = bytes

    def __tensor__(self, shape, dtype):
        channels = None
        if shape is not None:
            shape_len = len(shape)
            if shape_len < 3 or shape_len > 4:
                raise Exception("Unsupported shape: %s" % shape)
            channels = shape[-1]

        contents = as_tensor(self.bytes, None, None)
        image = tf.image.decode_image(contents, channels=channels)
        if dtype is not None and dtype != image.dtype:
            image = tf.image.convert_image_dtype(image, dtype=dtype)

        h = shape[-3]
        w = shape[-2]
        if w is not None or h is not None:
            image = tf.image.resize_image_with_crop_or_pad(image, target_height=h, target_width=w)

        return image

class _FileBytes:
    def __init__(self, path):
        self._path = path

    def __tensor__(self, shape, dtype):
        return tf.read_file(self._path)

def file(path):
    mimetype, _ = mimetypes.guess_type("file://%s" % path)
    return _decode(mimetype, _FileBytes(path))

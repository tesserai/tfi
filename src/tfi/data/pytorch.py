import mimetypes

from tfi.as_tensor import as_tensor as _as_tensor
from tfi.as_tensor import from_tensor as _from_tensor
from tfi.as_tensor import ShapeMismatchException as _ShapeMismatchException

from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision
import collections

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

def _resize_img(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    # if not _is_pil_image(img):
    #     raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

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
            channels = shape[0]
            h = shape[1]
            w = shape[2]

        with self.bytes.open() as f:
            with Image.open(f) as img:
                rgbimg = img.convert('RGB')

                if w is not None or h is not None:
                    rgbimg = _resize_img(rgbimg, (h, w), Image.BILINEAR)

                return torchvision.transforms.ToTensor()(rgbimg)

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
    return _from_tensor(
            tensor,
            [
                (dtype, shape, _compose(accept_mimetypes[mimetype], encoder))
                for mimetype, dtype, shape, encoder in _ENCODERS
                if mimetype in accept_mimetypes
            ])

# @_register_encoder(
#         ["image/png"],
#         [tf.uint8],
#         [(None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
# def _png_encode(tensor):
#     with tf.Graph().as_default() as g:
#         with tf.Session(graph=g):
#             # TODO(adamb) Use placeholder and a cached graph, for speed.
#             encoded = tf.image.encode_png(tensor)
#             return encoded.eval()
#
# @_register_encoder(
#         ["image/png"],
#         [tf.float32],
#         [(None, None, 1), (None, None, 2), (None, None, 3), (None, None, 4)])
# def _png_float_encode(tensor):
#     with tf.Graph().as_default() as g:
#         with tf.Session(graph=g):
#             # TODO(adamb) Use placeholder and a cached graph, for speed.
#             encoded = tf.image.encode_png(tf.cast(tensor * 255.0, tf.uint8))
#             return encoded.eval()

class _FileBytes(Decoder):
    def __init__(self, path):
        self._path = path

    def open(self):
        return open(self._path, 'rb')

    def __tensor__(self, shape, dtype):
        with open(self._path, 'rb') as f:
            return torch.ByteTensor(f.read())

from contextlib import contextmanager

class _OpenFile(Decoder):
    def __init__(self, f):
        self._f = f

    @contextmanager
    def open(self):
        yield self._f

def file(path, mimetype=None):
    if mimetype is None:
        mimetype, _ = mimetypes.guess_type("file://%s" % path)
    if isinstance(path, str):
        return _decode(mimetype, _FileBytes(path))

    return _decode(mimetype, _OpenFile(path))

import pprint
import sys
from tfi.data.terminal import imgcat

def terminal_write(o):
    tensor = _as_tensor(o, None, None)
    accept_mimetypes = {"image/png": imgcat, "text/plain": lambda x: x}
    encoded = _encode(tensor, accept_mimetypes)
    if encoded is None:
        encoded = tensor
    print(pprint.pformat(encoded), file=sys.stdout, flush=True)

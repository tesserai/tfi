import torch
import mimetypes

from tfi.tensor.codec import ShapeMismatchException as _ShapeMismatchException
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

def _decode(mimetype, *args):
    func = _DECODERS[mimetype]
    return func(*args)

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
def _decode_image(shape, dtype, f):
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

    with Image.open(f) as img:
        rgbimg = img.convert('RGB')

        if w is not None or h is not None:
            rgbimg = _resize_img(rgbimg, (h, w), Image.BILINEAR)

        return torchvision.transforms.ToTensor()(rgbimg)

class _driver(object):
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

    def _decode_open_file(self, shape, dtype, mimetype, handle):
        return _decode(mimetype, shape, dtype, handle)

    def _decode_file_path(self, shape, dtype, mimetype, path):
        if path.startswith("//"):
            path = "/Users/adamb/github/tesserai/tfi/src/tfi/data/%s" % path[2:]
        with open(path, 'rb') as f:
            return _decode(mimetype, shape, dtype, f)

    def _decode_bytes(self, shape, dtype, mimetype, bytes):
        return _decode(mimetype, shape, dtype, bytes)

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

from tfi.tensor.codec import _BaseAdapter

_as_tensor_adapter = _BaseAdapter(_driver())
as_tensor = _as_tensor_adapter.as_tensor
maybe_as_tensor = _as_tensor_adapter.maybe_as_tensor


import numpy as np
import torch
import torch.autograd
from torchvision import transforms

from io import BytesIO
from tfi.data import _FileSource

from tfi.tensor.codec import register_encoder as _register_encoder
from tfi.tensor.codec import register_tensor_spec as _register_tensor_spec

@_register_tensor_spec(_FileSource)
def _file_source_tensor_spec(fp):
    tensor = as_tensor(fp, None, None)
    return tensor, tensor.shape, type(tensor), np.reshape

@_register_tensor_spec(torch.autograd.variable.Variable)
def _variable_tensor_spec(v):
    return v.data, v.data.shape, type(v.data), np.reshape

@_register_encoder(
        ["python/jsonable"],
        [torch.FloatTensor, torch.IntTensor, torch.ShortTensor],
        [None])
def _jsonable_encode(tensor):
    return tensor.tolist()

@_register_encoder(
        ["image/png"],
        [torch.FloatTensor, torch.IntTensor, torch.ShortTensor],
        [(4, None, None), (3, None, None), (1, None, None)])
def _png_encode(tensor):
    s = BytesIO()
    transforms.ToPILImage()(tensor).save(s, 'png')
    return s.getvalue()

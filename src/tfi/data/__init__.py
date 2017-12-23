import mimetypes
import os.path

from io import BytesIO as _BytesIO

class _FileSource(object):
    """
    A serializable, "tensorizable" wrapper for a file source. Serialization
    reads from the file and embeds the bytes into the state dict.
    """
    def __init__(self, mimetype=None, tensor_fn=None, bytes_fn=None):
        self.mimetype = mimetype
        self._tensor_fn = tensor_fn
        self._bytes_fn = bytes_fn

    def __tensor__(self, ops, shape, dtype):
        return self._tensor_fn(ops, shape, dtype, self.mimetype)

    def __setstate__(self, d):
        self.mimetype = d['mimetype']
        self._tensor_fn = lambda ops, shape, dtype, mimetype: ops.decode_open_file(shape, dtype, mimetype, _BytesIO(d['bytes']))
        self._bytes_fn = lambda: d['bytes']

    def __getstate__(self):
        return {
            'mimetype': self.mimetype,
            'bytes': self._bytes_fn(),
        }

def _read_file_path(path):
    with open(path, "rb") as f:
        return f.read()

def _read_file_handle(f):
    try:
        pos = f.tell()
        return f.read()
    finally:
        f.seek(pos)

def file(arg, mimetype=None):
    if arg.startswith("//"):
        arg = os.path.join(os.path.dirname(__file__), arg[2:])

    if isinstance(arg, str):
        if mimetype is None:
            mimetype, _ = mimetypes.guess_type("file://%s" % arg)
        tensor_fn = lambda ops, shape, dtype, mimetype: ops.decode_file_path(shape, dtype, mimetype, arg)
        bytes_fn = lambda: _read_file_path(arg)
    else:
        tensor_fn = lambda ops, shape, dtype, mimetype: ops.decode_open_file(shape, dtype, mimetype, arg)
        bytes_fn = lambda: _read_file_handle(arg)

    return _FileSource(mimetype, tensor_fn, bytes_fn)

import mimetypes
import os.path
import urllib.request

from io import BytesIO as _BytesIO

class _FileSource(object):
    """
    A serializable, "tensorizable" wrapper for a file source. Serialization
    reads from the file and embeds the bytes into the state dict.
    """
    def __init__(self, mimetype=None, tensor_fn=None, read_fn=None, repr_bytes_fn=None):
        self.mimetype = mimetype
        self._tensor_fn = tensor_fn
        self._read_fn = read_fn
        self._repr_bytes_fn = repr_bytes_fn

    def __tensor__(self, ops, shape, dtype):
        return self._tensor_fn(ops, shape, dtype, self.mimetype)

    def __setstate__(self, d):
        self.mimetype = d['mimetype']
        self._tensor_fn = lambda ops, shape, dtype, mimetype: ops.decode_open_file(shape, dtype, mimetype, _BytesIO(d['bytes']))
        self._read_fn = lambda: d['bytes']
        self._repr_bytes_fn = lambda: ('base64', _base64.b64encode(d['bytes']), d['mimetype'])

    def read(self):
        return self._read_fn()

    def __getstate__(self):
        return {
            'mimetype': self.mimetype,
            'bytes': self._read_fn(),
        }

    def __repr__(self):
        t, d, m = self._repr_bytes_fn()
        mimetype_repr = ""
        if m is not None:
            mimetype_repr = ", mimetype=%s" % repr(m)
        return "tfi.data.%s(%r%s)" % (t, d, mimetype_repr)

    def __json__(self):
        t, d, m = self._repr_bytes_fn()
        di = {}
        if t == 'file':
            di['$ref'] = d
        elif t == 'base64':
            di['$base64'] = d

        if m is not None:
            di['$mimetype'] = m

        return di

def _with_urlopen(url, fn):
    with urllib.request.urlopen(url) as f:
        return fn(f)

def _read_file_path(path):
    with open(path, "rb") as f:
        return f.read()

def _read_url(url):
    with urllib.request.urlopen(url) as f:
        return f.read()

def _read_file_handle(f):
    pos = None
    try:
        pos = f.tell()
        return f.read()
    finally:
        if pos is not None:
            f.seek(pos)

import json as _json
import base64 as _base64
from tfi.base import _recursive_transform

def json(s):
    def _replace_ref(v):
        if not isinstance(v, dict) or '$ref' not in v:
            return v
        ref = v['$ref']
        if ref.startswith('http://') or ref.startswith('https://') or ref.startswith('//'):
            return file(ref)
        return v

    v = _json.loads(s)
    # Support referencing HTTP and HTTPS URLs with {"$ref":"https://..."}
    return _recursive_transform(v, _replace_ref)

def base64(arg, mimetype=None):
    repr_bytes_fn = lambda: ('base64', arg, mimetype)
    read_fn = lambda: _base64.b64decode(arg)
    tensor_fn = lambda ops, shape, dtype, mimetype: ops.decode_open_file(shape, dtype, mimetype, _BytesIO(read_fn()))

    return _FileSource(mimetype, tensor_fn, read_fn, repr_bytes_fn)

def file(arg, mimetype=None):
    repr_bytes_fn = lambda: ('file', arg, mimetype)
    _mimetype = mimetype
    _arg = arg

    if isinstance(_arg, str):
        if _arg.startswith("//"):
            _arg = os.path.join(os.path.dirname(__file__), _arg[2:])

        if _arg.startswith("http://") or _arg.startswith("https://"):
            url = _arg
            tensor_fn = lambda ops, shape, dtype, _mimetype: _with_urlopen(_arg, lambda f: ops.decode_open_file(shape, dtype, _mimetype, f))
            read_fn = lambda: _read_url(_arg)
        else:
            url = "file://%s" % _arg
            tensor_fn = lambda ops, shape, dtype, _mimetype: ops.decode_file_path(shape, dtype, _mimetype, _arg)
            read_fn = lambda: _read_file_path(_arg)

        if _mimetype is None:
            _mimetype, _ = mimetypes.guess_type(url)

    else:
        tensor_fn = lambda ops, shape, dtype, _mimetype: ops.decode_open_file(shape, dtype, _mimetype, _arg)
        read_fn = lambda: _read_file_handle(arg)

    return _FileSource(_mimetype, tensor_fn, read_fn, repr_bytes_fn)

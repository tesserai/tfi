import mimetypes as _mimetypes
import os.path
import urllib.request

import json as _json
import base64 as _base64
from tfi.base import _recursive_transform

from io import BytesIO as _BytesIO

# class _Source(object):
#     """
#     A serializable, "tensorizable" wrapper for a file source. Serialization
#     reads from the file and embeds the bytes into the state dict.
#     """
#     def __init__(self, mimetype=None, repr_fn=None):
#         self.mimetype = mimetype
#         self._repr_fn = repr_fn

#     def __repr__(self):
#         t, d, m = self._repr_fn()
#         mimetype_repr = ""
#         if m is not None:
#             mimetype_repr = ", mimetype=%s" % repr(m)
#         return "tfi.data.%s(%r%s)" % (t, d, mimetype_repr)

#     def read(self):
#         t, d, m = self._repr_fn()
#         if t == 'base64':
#             return _base64.b64decode(d)
#         if t == 'url':
#             with urllib.request.urlopen(d) as f:
#                 return f.read()
#         if t == 'path':
#             with open(d, "rb") as f:
#                 return f.read()
#         if t == 'stream':
#             pos = None
#             try:
#                 pos = d.tell()
#                 return d.read()
#             finally:
#                 if pos is not None:
#                     d.seek(pos)


#     def __tensor__(self, ops, shape, dtype):
#         t, d, m = self._repr_fn()
#         if t == 'path':
#             return ops.decode_file_path(shape, dtype, m, d)
#         if t == 'base64':
#             return ops.decode_open_file(shape, dtype, m, _BytesIO(_base64.b64decode(d)))
#         if t == 'url':
#             with urllib.request.urlopen(d) as f:
#                 return ops.decode_open_file(shape, dtype, m, f)
#         return ops.decode_open_file(shape, dtype, m, d)

#     def __setstate__(self, d):
#         print("d", d.keys())
#         if 'bytes' in d:
#             d = {
#                 'type': 'base64',
#                 'mimetype': d['mimetype'],
#                 'data': _base64.b64encode(d['bytes']),
#             }
#         print("d", d.keys())
#         self.mimetype = d['mimetype']
#         self._repr_fn = lambda: (d['type'], d['data'], d['mimetype'])

#     def __getstate__(self):
#         t, d, m = self._repr_fn()
#         return {
#             'mimetype': m,
#             'type': t,
#             'data': d,
#         }

#     def __json__(self):
#         t, d, m = self._repr_fn()

#         r = {}
#         if t == 'url' or t == 'path':
#             r['$ref'] = d
#         elif t == 'base64':
#             r['$base64'] = d

#         if m is not None:
#             r['$mimetype'] = m

#         print("__json__", r.keys())
#         return r

class _Source(object):
    def __repr__(self):
        return "tfi.data.%s(%s)" % (self.__class__.__name__, ", ".join([repr(arg) for arg in self._repr_args()]))

class base64(_Source):
    def __init__(encoded=None, mimetype=None):
        """Returns a tensor-adaptable representation of the bytes in the given base64 encoded string"""
        self._encoded = encoded
        self._mimetype = mimetype
 
    def __setstate__(self, d):
        self._encoded = d['$base64']
        self._mimetype = d.get('$mimetype', None)
    
    def _repr_args(self):
        return (self._encoded,) if self._mimetype is None else (self._encoded, self._mimetype)

    def __getstate__(self):
        return self.__json__()

    def __json__(self):
        r = {}
        r['$base64'] = self._encoded
        if self._mimetype is not None:
            r['$mimetype'] = self._mimetype
        return r

    def read(self):
        return _base64.b64decode(self._encoded)

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_open_file(shape, dtype, self._mimetype, _BytesIO(_base64.b64decode(self._encoded)))

def idempotent(fn):
    cached = None
    def lazy(self):
        nonlocal cached
        if cached is None:
            cached = fn(self)
        return cached
    lazy.__wrapped__ = fn
    return lazy

class url(_Source):
    def __init__(self, url=None, mimetype=None):
        """Returns a tensor-adaptable representation of the HTTP response body returned after an HTTP GET of the given URL"""
        if url is not None and not url.startswith("http://") and not url.startswith("https://"):
            raise Exception("Given URL must start with http:// or https://")

        self._url = url
        self._mimetype = mimetype
    
    def _repr_args(self):
        return (self._url,) if self._mimetype is None else (self._url, self._mimetype)

    @idempotent
    def _resolve_mimetype(self):
        return self._mimetype or _mimetypes.guess_type(self._url)[0]

    def __setstate__(self, d):
        self._url = d['$ref']
        self._mimetype = d.get('$mimetype', None)

    def __getstate__(self):
        return self.__json__()

    def __json__(self):
        r = {}
        r['$ref'] = self._url
        if self._mimetype is not None:
            r['$mimetype'] = self._mimetype
        return r

    def read(self):
        with urllib.request.urlopen(self._url) as f:
            return f.read()

    def __tensor__(self, ops, shape, dtype):
        with urllib.request.urlopen(self._url) as f:
            return ops.decode_open_file(shape, dtype, self._resolve_mimetype(), f)

class path(_Source):
    def __init__(self, path=None, mimetype=None):
        """Returns a tensor-adaptable representation of the file at the given path"""
        self._path = path
        self._mimetype = mimetype

    def _repr_args(self):
        return (self._path,) if self._mimetype is None else (self._path, self._mimetype)

    @idempotent
    def _resolve_mimetype(self):
        return self._mimetype or _mimetypes.guess_type("file://%s" % self._resolve_path())[0]

    def _resolve_path(self):
        if self._path.startswith("//"):
            return os.path.join(os.path.dirname(__file__), self._path[2:])
        return self._path

    def __setstate__(self, d):
        self._path = d['$ref']
        self._mimetype = d.get('$mimetype', None)

    def __getstate__(self):
        return self.__json__()

    def __json__(self):
        r = {}
        r['$ref'] = self._path
        if self._mimetype is not None:
            r['$mimetype'] = self._mimetype
        return r

    def read(self):
        with open(self._resolve_path(), 'rb') as f:
            return f.read()

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_file_path(shape, dtype, self._resolve_mimetype(), self._resolve_path())

# class stream(_Source):
#     def __init__(self, _arg, _mimetype):
#         """Returns a tensor-adaptable representation of the bytes at current position of the given stream"""
#         self._stream = stream
#         self._mimetype = _mimetype

#     def read(self):
#         pos = None
#         try:
#             pos = d.tell()
#             return d.read()
#         finally:
#             if pos is not None:
#                 d.seek(pos)
    
#     def __setstate__(self, d):
#         pass

#     def __json__(self):
#         pass

#     def __tensor__(self, ops, shape, dtype):
#         return ops.decode_open_file(shape, dtype, self._mimetype, self._stream)

def file(arg, mimetype=None):
    if isinstance(arg, str):
        if arg.startswith("http://") or arg.startswith("https://"):
            return url(arg, mimetype)
        return path(arg, mimetype)
    return stream(arg, mimetype)

def json(s):
    """Returns a tensor-adaptable representation of the contents of a JSON-formatted string"""
    def _reify_ref_and_base64(v):
        if not isinstance(v, dict):
            return v
        if '$ref' in v:
            ref = v['$ref']
            if ref.startswith('http://') or ref.startswith('https://') or ref.startswith('//'):
                return file(ref, v.get('$mimetype', None))
            return v
        if '$base64' in v:
            return base64(v['$base64'], v.get('$mimetype', None))
        return v

    v = _json.loads(s)
    # Support:
    # - referencing HTTP and HTTPS URLs with {"$ref":"https://..."}
    # - byte arrays with {"$base64": "Q423z..."}
    return _recursive_transform(v, _reify_ref_and_base64)


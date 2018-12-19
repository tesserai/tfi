import mimetypes as _mimetypes
import os.path
import urllib.request

import tfi.json as _tfi_json
import base64 as _base64
from tfi.base import _recursive_transform

from io import BytesIO as _BytesIO

class _Source(object):
    def __repr__(self):
        return "tfi.data.%s(%s)" % (self.__class__.__name__, ", ".join([repr(arg) for arg in self._repr_args()]))

class _jsonable_bytes(bytes):
    def __new__(cls, val, jsonfn=None, fetchablefn=None):
        r = super().__new__(cls, val)
        r._jsonfn = jsonfn
        r._fetchablefn = fetchablefn
        return r

    def __fetchable__(self):
        return self._fetchablefn()

    def __json__(self):
        return self._jsonfn()

class base64(_Source):
    def __init__(self, encoded=None, mimetype=None):
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

    def open(self):
        return _BytesIO(self.read())

    def read(self):
        return _base64.b64decode(self._encoded)

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_bytes(shape, dtype, self._mimetype, _base64.b64decode(self._encoded))
        

class bytes(_Source):
    def __init__(self, bytes=None, mimetype=None):
        """Returns a tensor-adaptable representation of the given bytes"""
        self._bytes = bytes
        self._mimetype = mimetype
 
    def __setstate__(self, d):
        self._bytes = d['$bytes']
        self._mimetype = d.get('$mimetype', None)
    
    def _repr_args(self):
        return (self._bytes,) if self._mimetype is None else (self._bytes, self._mimetype)

    def __getstate__(self):
        return self.__json__()

    def __json__(self):
        r = {}
        r['$bytes'] = self._bytes
        if self._mimetype is not None:
            r['$mimetype'] = self._mimetype
        return r

    def open(self):
        return _BytesIO(self._bytes)

    def read(self):
        return self._bytes

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_bytes(shape, dtype, self._mimetype, self._bytes)

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
    def read_mimetype(self):
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

    def open(self):
        return urllib.request.urlopen(self._url)

    def read(self):
        with urllib.request.urlopen(self._url) as f:
            return f.read()

    def __tensor__(self, ops, shape, dtype):
        with urllib.request.urlopen(self._url) as f:
            return ops.decode_open_file(shape, dtype, self.read_mimetype(), f)

class path(_Source):
    def __init__(self, path=None, mimetype=None):
        """Returns a tensor-adaptable representation of the file at the given path"""
        self._path = path
        self._mimetype = mimetype

    def _repr_args(self):
        return (self._path,) if self._mimetype is None else (self._path, self._mimetype)

    @idempotent
    def read_mimetype(self):
        return self._mimetype or _mimetypes.guess_type("file://%s" % self._resolve_path())[0]

    def _resolve_path(self):
        if self._path.startswith("tfi://"):
            return os.path.join(os.path.dirname(__file__), self._path[6:])
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

    def open(self):
        return open(self._resolve_path(), 'rb')

    def read(self):
        with open(self._resolve_path(), 'rb') as f:
            return f.read()

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_file_path(shape, dtype, self.read_mimetype(), self._resolve_path())

class assets_extra(path):
    def __init__(self, mimetype=None, assets_extra_path=None, assets_extra_root=None):
        path = os.path.join(assets_extra_root, assets_extra_path)
        super().__init__(path=path, mimetype=mimetype)
        self._assets_extra_root = assets_extra_root
        self._assets_extra_path = assets_extra_path

    def assets_extra_root(self):
        return self._assets_extra_root

    def assets_extra_path(self):
        return self._assets_extra_path

    def __fetchable__(self):
        return {
            'basename': os.path.basename(self._assets_extra_path),
            'urlpath': self._assets_extra_path,
            'mimetype': self._mimetype,
        }

class stream(_Source):
    def __init__(self, arg, mimetype):
        """Returns a tensor-adaptable representation of the bytes at current position of the given stream"""
        self._stream = arg
        self._mimetype = mimetype

    def read_mimetype(self):
        return self._mimetype

    def __repr__(self):
        return "tfi.data.base64(%r%s)" % (self._as_base64(), "" if self._mimetype is None else ", mimetype=%s" % self._mimetype)

    def open(self):
        return _BytesIO(self.read())

    def read(self):
        if not hasattr(self._stream, 'tell'):
            print("no tell for", self._stream, "converting to BytesIO")
            b = self._stream.read()
            self._stream = _BytesIO(b)

        pos = None
        try:
            pos = self._stream.tell()
            return self._stream.read()
        finally:
            if pos is not None:
                self._stream.seek(pos)

    def _as_base64(self):
        return _base64.b64encode(self.read())
    
    def __setstate__(self, d):
        self._stream = _BytesIO(_base64.b64decode(d['$base64']))
        self._mimetype = d.get('$mimetype', None)

    def __getstate__(self):
        return self.__json__()

    def __json__(self):
        r = {}
        r['$base64'] = self.read()
        if self._mimetype is not None:
            r['$mimetype'] = self._mimetype
        return r

    def __tensor__(self, ops, shape, dtype):
        return ops.decode_open_file(shape, dtype, self._mimetype, self._stream)

def file(arg, mimetype=None):
    if isinstance(arg, str):
        if arg.startswith("http://") or arg.startswith("https://"):
            return url(arg, mimetype)
        return path(arg, mimetype)
    return stream(arg, mimetype)

def json(s, assets_extra_root=None):
    """Returns a tensor-adaptable representation of the contents of a JSON-formatted string"""
    def _reify_ref_and_base64(v):
        if not isinstance(v, dict):
            return v

        if '$ref' in v:
            v_orig = v
            ref = v_orig['$ref']
            if ref.startswith('http://') or ref.startswith('https://') or ref.startswith('tfi://'):
                v = file(ref, mimetype=v.get('$mimetype', None))
            elif assets_extra_root is not None and ref.startswith("assets.extra://"):
                assets_extra_path = ref[len("assets.extra://"):]
                v = assets_extra(
                    assets_extra_path=assets_extra_path,
                    assets_extra_root=assets_extra_root,
                    mimetype=v.get('$mimetype', None),
                )
            else:
                return v_orig

            # # HACK(adamb) Don't keep this!!!
            # if ref.startswith('.') or ref.startswith('/'):
            #     v = file(ref, v.get('$mimetype', None))

            if '$encode' in v_orig and v_orig['$encode']:
                encoding = v_orig['$encode']
                if encoding != 'base64':
                    raise Exception("Unknown value for $encode: \"%s\"" % encoding)

                return _jsonable_bytes(
                    _base64.b64encode(v.read()),
                    jsonfn=lambda: v_orig,
                    fetchablefn=v.__fetchable__,
                )
            return v
        elif '$base64' in v:
            return base64(v['$base64'], v.get('$mimetype', None))

        return v

    v = _tfi_json.loads(s)
    # Support:
    # - referencing HTTP and HTTPS URLs with {"$ref":"https://..."}
    # - byte arrays with {"$base64": "Q423z..."}
    return _recursive_transform(v, _reify_ref_and_base64)


from tfi.tensor.frame import TensorFrame as _TensorFrame
from tfi.tensor.codec import encode as _tensor_encode
from tfi.base import _recursive_transform

import base64
import numpy as np

_ACCEPT_MIMETYPES = {
  # "image/png": lambda x: base64.b64encode(x),
  "image/png": lambda x: x,
  "text/plain": lambda x: x,
  # Use python/jsonable so we to a recursive transform before jsonification.
  "python/jsonable": lambda x: x,
}

_TRANSFORMS = {}

def _transform_value(o):
  t = type(o)
  if t in _TRANSFORMS:
    o = _TRANSFORMS[t](o)

  return _tensor_encode(_ACCEPT_MIMETYPES, o)


def as_jsonable(result):
  r = _recursive_transform(result, _transform_value)
  if r is not None:
    return r
  return result


_TRANSFORMS[_TensorFrame] = lambda o: _TensorFrame(
      *[
        (shape, name, _recursive_transform(tensor, _transform_value))
        for shape, name, tensor in o.tuples()
      ],
      **o.shape_labels(),
    ).zipped(jsonable=True),

_TRANSFORMS[np.int32] = lambda o: int(o)

try:
  import pandas as _pandas

  def df_to_dict(df):
    d = df.to_dict('split')
    del d['index']
    return d
  _TRANSFORMS[_pandas.DataFrame] = df_to_dict
except ImportError:
  pass

try:
  import pyarrow as _pyarrow
  _TRANSFORMS[_pyarrow.lib.Buffer] = lambda o: base64.b64encode(o.to_pybytes()).decode('utf-8')
except ImportError:
  pass

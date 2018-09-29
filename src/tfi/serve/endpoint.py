import json
import inspect
import opentracing

import tfi.tensor.codec

from flask import request, jsonify, make_response
from opentracing.ext import tags as opentracing_ext_tags
from urllib.parse import urlparse
from werkzeug.exceptions import BadRequest

from tfi import data as tfi_data
from tfi.base import _recursive_transform

def _replace_ref(v):
  if not isinstance(v, dict) or '$ref' not in v:
    return v
  ref = v['$ref']
  if ref.startswith('http://') or ref.startswith('https://'):
    return tfi_data.file(ref)
  return v

def _field(req, field, annotation):
  if field in req.files:
    file = req.files[field]
    return True, tfi_data.file(file, mimetype=file.mimetype)
  if field in req.form:
    v = req.form[field]
    # TODO(adamb) Better to use some kind of "from_string" entry
    if isinstance(annotation, dict) and 'dtype' in annotation:
      v = annotation['dtype'](v)
    else:
      v = tfi_data.json(v)
    return True, v

  # TODO(adamb) Test!
  json_data = req.get_json()
  if json_data:
    return (field in json_data, json_data.get(field, None))
  return False, None

def _default_if_empty(v, default):
  return v if v is not inspect.Parameter.empty else default

def _maybe_plural(n, singular, plural):
  return singular if n == 1 else plural

from tfi.tensor.frame import TensorFrame as _TensorFrame

# TODO(adamb) Add tracing to /specialize
# TODO(adamb) Switch _field logic to transforming to a "json" object.
#   map uploaded files to base64-encoded files. Fix tensor mapping
#   to "detect" base64 encodings for png, jpg files.


def _MissingParameters(missing):
  noun = _maybe_plural(len(missing), "parameter", "parameters")
  desc = "Missing %s: %s" % (noun, ", ".join(missing))
  return BadRequest(desc)

def make_endpoint(model, method_name):
  method = getattr(model, method_name)
  sig = inspect.signature(method)
  param_annotations = {k: v.annotation for k, v in sig.parameters.items()}
  required = {k for k, v in sig.parameters.items() if v.default is inspect.Parameter.empty}

  accept_mimetypes = {
    # "image/png": lambda x: base64.b64encode(x),
    "image/png": lambda x: x,
    "text/plain": lambda x: x,
    # Use python/jsonable so we to a recursive transform before jsonification.
    "python/jsonable": lambda x: x,
  }

  def _transform_value(o):
    if isinstance(o, _TensorFrame):
      o = _TensorFrame(
        *[
          (shape, name, _recursive_transform(tensor, _transform_value))
          for shape, name, tensor in o.tuples()
        ],
        **o.shape_labels(),
      ).zipped(jsonable=True)

    return tfi.tensor.codec.encode(accept_mimetypes, o)

  def fn():
    d = {}
    missing = set(required)
    for k, ann in param_annotations.items():
      ok, v = _field(request, k, ann)
      if ok:
        missing.remove(k)
        d[k] = v

    if missing:
      raise _MissingParameters(missing)
    
    result = method(**d)

    r = _recursive_transform(result, _transform_value)
    if r is not None:
      result = r
    
    response = make_response(jsonify(result), 200)

    return response
  return fn

import base64
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

from collections import OrderedDict


def _decode_request(req):
  if req.form is not None:
    decoded = OrderedDict()
    for field, value in req.form.items():
      ch = value[0]
      if ch != '{' and ch != '[' and ch != '"' and not value.isdecimal():
        # HACK(adamb) Replace value with a json-encoded version of itself
        value = json.dumps(value)
      decoded[field] = tfi_data.json(value)

    for field, file in req.files.items():
      decoded[field] = tfi_data.file(file, mimetype=file.mimetype)
    return decoded
  
  if req.get_json() is not None:
    return req.get_json()

def _get_request_field(req, field, annotation):
  if field not in req:
    return False, None
  
  return True, req[field]

def _maybe_plural(n, singular, plural):
  return singular if n == 1 else plural

from tfi.json import as_jsonable as _as_jsonable

def _MissingParameters(missing):
  noun = _maybe_plural(len(missing), "parameter", "parameters")
  desc = "Missing %s: %s" % (noun, ", ".join(missing))
  return BadRequest(desc)

def make_endpoint(model, method_name):
  method = getattr(model, method_name)
  sig = inspect.signature(method)
  param_annotations = {k: v.annotation for k, v in sig.parameters.items()}
  required = {k for k, v in sig.parameters.items() if v.default is inspect.Parameter.empty}

  def fn():
    decoded = _decode_request(request)
    d = {}
    missing = set(required)
    for k, ann in param_annotations.items():
      ok, v = _get_request_field(decoded, k, ann)
      if ok:
        v = decoded[k]
        missing.remove(k)
        d[k] = v

    if missing:
      raise _MissingParameters(missing)
    
    result = method(**d)
    jsonable = _as_jsonable(result)
    
    response = make_response(jsonify(jsonable), 200)

    return response
  return fn

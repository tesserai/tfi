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

from tfi.tensor.codec import encode as _encode_tensor

def _decode_form_value(value):
  # Attempt to auto-detect JSON-encoded values
  ch = value[0]
  # Assume that values that start with a char that is 
  # - non-object,
  # - non-array,
  # - non-quote,
  # - and non-numeric
  # should be considered a string.
  if ch != '{' and ch != '[' and ch != '"' and not value.isdecimal():
    # HACK(adamb) Replace value with a json-encoded version of itself
    value = json.dumps(value)
  return tfi_data.json(value)


def _decode_request(request):
  print("_decode_request", request, request.mimetype)

  decoded = OrderedDict()

  if 'x-tesserai-input-json' in request.headers:
    decoded.update(json.loads(request.headers['x-tesserai-input-json']))

  if 'x-tesserai-input-body' in request.headers:
    body_key = request.headers['x-tesserai-input-body']
    decoded[body_key] = tfi_data.bytes(request.data, mimetype=request.mimetype)
    return decoded

  decoded_json = request.get_json()
  if decoded_json is not None:
    decoded.update(decoded_json)

  if request.form is not None:
    for field in request.form:
      values = request.form.getlist(field)
      decoded[field] = [
        _decode_form_value(value)
        for value in values
      ]

      if len(decoded[field]) == 1:
        decoded[field] = decoded[field][0]

    for field, file in request.files.items():
      print("field, file", field, file)
      decoded[field] = tfi_data.file(file, mimetype=file.mimetype)
  
  return decoded

from tfi.json import as_jsonable as _as_jsonable

def _encode_response(request, result):
  if 'x-tesserai-output-body' in request.headers:
    body_key = request.headers['x-tesserai-output-body']
    if body_key[0] != '[':
      body_key = [body_key]
    else:
      body_key = json.loads(body_key)

    for key in body_key:
      result = result[key]

  accept = request.headers['accept']
  if '*/*' in accept or 'application/json' in accept:
    jsonable = _as_jsonable(result)
    return 'application/json', jsonify(jsonable)

  return accept, _encode_tensor({accept: lambda x: x}, result)

def _get_request_field(req, field, annotation):
  if field not in req:
    return False, None
  
  return True, req[field]

def _maybe_plural(n, singular, plural):
  return singular if n == 1 else plural

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
    mimetype, encoded = _encode_response(request, result)
    response = make_response(encoded, 200)
    response.headers['Content-Type'] = mimetype
    return response

  return fn

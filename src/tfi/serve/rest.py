import base64
import json
import inspect
import opentracing

import tfi.tensor.codec

from flask import request, make_response
from opentracing.ext import tags as opentracing_ext_tags
from urllib.parse import urlparse
from werkzeug.exceptions import BadRequest

from tfi import data as tfi_data
from tfi import json as tfi_json
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

def _decode_predict_pb2_request(request):
  decoded = OrderedDict()
  from tensorflow.python.framework import tensor_util
  import tensorflow_serving_api.predict_pb2
  pr = tensorflow_serving_api.predict_pb2.PredictRequest()
  pr.ParseFromString(request.data)
  for input_name, input_tensor_proto in pr.inputs.items():
    decoded[input_name] = tensor_util.MakeNdarray(input_tensor_proto)
  return list(pr.output_filter), decoded

def _decode_request(request):
  # TODO(adamb) Evaluate whether or not request is a tf.PredictionRequest
  # If so, decode it using tensorflow-serving-api
  if request.mimetype == 'application/x-protobuf':
    return _decode_predict_pb2_request(request)

  output_filter = None
  decoded = OrderedDict()

  if 'x-tesserai-input-json' in request.headers:
    decoded.update(json.loads(request.headers['x-tesserai-input-json']))

  if 'x-tesserai-input-body' in request.headers:
    body_key = request.headers['x-tesserai-input-body']
    decoded[body_key] = tfi_data.bytes(request.data, mimetype=request.mimetype)
    return output_filter, decoded

  if request.mimetype == 'application/json':
    decoded_json = tfi_data.json(request.data)
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
      decoded[field] = tfi_data.file(file, mimetype=file.mimetype)
  
  return output_filter, decoded

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
  if '*/*' in accept:
    if request.mimetype == 'application/json':
      accept = 'application/json'
    elif request.mimetype == 'application/x-protobuf':
      accept = 'application/x-protobuf'

  # NOTE(adamb) This is a bit gross and confusing. Why can't
  #     we always use _encode_tensor?
  if 'application/json' in accept:
    return 'application/json', tfi_json.dumps(result, coerce=True)

  if 'application/x-protobuf' in accept:
    # TODO(adamb) Properly encode outputs
    return 'application/x-protobuf', None

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

def _make_method_endpoint(model, method_name):
  method = getattr(model, method_name)
  sig = inspect.signature(method)
  param_annotations = {k: v.annotation for k, v in sig.parameters.items()}
  required = {k for k, v in sig.parameters.items() if v.default is inspect.Parameter.empty}

  def fn():
    output_filter, decoded = _decode_request(request)
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
    
    # TODO(adamb) Support for output_filter, so we can avoid computing outputs
    #     we don't care about.
    result = method(**d)
    mimetype, encoded = _encode_response(request, result)
    response = make_response(encoded, 200)
    response.headers['Content-Type'] = mimetype
    return response

  return fn

def _set_environ_later(k, v):
  def _wrap(f):
    def _do():
      request.environ[k] = v
      return f()
    return _do
  return _wrap

def add_endpoints(model, tracer, app):
  trace_route = tracer(app)

  for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
    if method_name.startswith('_'):
      continue

    tracing_tags = {
      "model.method": method_name,
      "visibility": "public",
    }
    fn = _make_method_endpoint(model, method_name)
    fn = _set_environ_later('TFI_METHOD', method_name)(fn)
    fn = trace_route(tracing_tags)(fn)
    fn.__name__ = method_name
    app.route("/api/%s" % method_name, methods=["POST", "GET"])(fn)


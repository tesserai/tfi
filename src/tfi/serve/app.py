import inspect
import json
import os.path
import urllib

from flask import Flask, request, send_file, make_response
from tfi.doc import documentation, render

from tfi.asset import asset_path as _asset_path

from tfi.serve.endpoint import make_endpoint as _make_endpoint

ERROR_ENCODER = json.JSONEncoder(sort_keys=True, indent=2, separators=(',', ': '))

def _make_json_error_response(error_obj, status_code):
  error_json = ERROR_ENCODER.encode(error_obj) + "\n"
  response = make_response(error_json, status_code, {'Content-Type': 'application/json'})
  return response

def _public_url(request, path, params='', query='', fragment=''):
  headers = request.headers
  return urllib.parse.ParseResult(
    scheme=headers.get('X-Forwarded-Proto', 'http'),
    netloc=headers.get('X-Forwarded-Host', headers.get('Host', '127.0.0.1')),
    path=path,
    params=params,
    query=query,
    fragment=fragment,
  ).geturl()

def _set_environ_later(k, v):
  def _wrap(f):
    def _do():
      request.environ[k] = v
      return f()
    return _do
  return _wrap

def make_app(model, tracer, model_file_fn=None, extra_scripts=""):
  if model is None:
    raise Exception("No model given")

  static_folder = os.path.abspath(
      os.path.join(
          os.path.dirname(os.path.dirname(__file__)),
          'static'))

  app = Flask(__name__,
      static_url_path="/static",
      static_folder=static_folder)

  trace_route = tracer(app)

  for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
    if method_name.startswith('_'):
      continue

    tracing_tags = {"model.method": method_name, "visibility": "public"}
    fn = _make_endpoint(model, method_name)
    fn = _set_environ_later('TFI_METHOD', method_name)(fn)
    fn = trace_route(tracing_tags)(fn)
    fn.__name__ = method_name
    app.route("/api/%s" % method_name, methods=["POST", "GET"])(fn)

  if model_file_fn:
    @app.route("/meta/snapshot", methods=["GET"])
    def meta_snapshot():
      # For now we assume that this is a read-only model, so
      # just return the codepath directly.
      return send_file(model_file_fn())

  @app.route("/object/<path:objectpath>", methods=["GET"])
  def get_object(objectpath):
    asset_path = _asset_path(model, objectpath)
    if asset_path is None:
      return make_response({"error": "Not found"}, 404)
    return send_file(asset_path)

  @app.route("/ok", methods=["GET"])
  def ok():
    return """{"status":"OK"}"""

  @app.route("/", methods=["GET"])
  def docs():
    doc_dict = documentation(model)
    headers = request.headers
    return render(**doc_dict,
        include_snapshot=model_file_fn is not None,
        proto=headers.get('X-Forwarded-Proto', 'http'),
        host=headers.get('X-Forwarded-Host', headers['HOST']),
        extra_scripts=extra_scripts)

    return response

  @app.errorhandler(400)
  @app.errorhandler(500)
  def make_error_response_from_exception(exception):
    status_code = exception.status_code if hasattr(exception, 'status_code') else 500

    doc_url = ""
    if 'TFI_METHOD' in request.environ:
      method_name = request.environ['TFI_METHOD']      
      doc_url = _public_url(request, "/", fragment="method-%s" % method_name)

    if hasattr(exception, 'description'):
      message = exception.description
    else:
      message = str(exception)

    error_obj = {
      "message": message,
    }
    if doc_url:
      error_obj["doc_url"] = doc_url

    return _make_json_error_response({"error": error_obj}, status_code)

  return app

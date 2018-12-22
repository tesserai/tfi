import inspect
import json
import os.path
import urllib

from flask import Flask, request, send_file, make_response, redirect, Response
from tfi.doc.template import HtmlRenderer

from tfi.asset import asset_path as _asset_path

from tfi.serve.rest import add_endpoints as _add_rest_endpoints


ERROR_ENCODER = json.JSONEncoder(sort_keys=True, indent=2, separators=(',', ': '))


class FixedLocationResponse(Response):
    autocorrect_location_header = False

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

def make_app(model, tracer, model_file_fn=None, extra_scripts=""):
  if model is None:
    raise Exception("No model given")

  static_folder = os.path.abspath(
      os.path.join(
          os.path.dirname(os.path.dirname(__file__)),
          'static'))

  app = Flask(__name__,
      static_url_path="/doc/en/static",
      static_folder=static_folder)

  _add_rest_endpoints(model, tracer, app)

  if model_file_fn:
    @app.route("/meta/snapshot", methods=["GET"])
    def meta_snapshot():
      # For now we assume that this is a read-only model, so
      # just return the codepath directly.
      return send_file(model_file_fn())

  # Subscribers to this model's event stream might be late.
  # provide a response that looks like the end of the upload
  # stream so they know to refresh.
  @app.route('/meta/events')
  def meta_events():    
    headers = {
	    'Cache-Control': 'no-cache',
	    'Connection': 'keep-alive',
      'x-envoy-upstream-rq-timeout-ms': '0',
    }
    return Response(
      """event: status
data: {"status":"done"}

""",
      200,
      content_type='text/event-stream',
      headers=headers,
    )

  @app.route("/object/<path:objectpath>", methods=["GET"])
  def get_object(objectpath):
    asset_path = _asset_path(model, "object/%s" % objectpath)
    if asset_path is None:
      return make_response(json.dumps({"error": "Not found", "path": objectpath}), 404)
    
    return send_file(asset_path)

  @app.route("/ok", methods=["GET"])
  def ok():
    return """{"status":"OK"}"""

  @app.route("/", methods=["GET"])
  def root():
    return redirect('/doc/en/', 302, Response=FixedLocationResponse)

  doc_renderer = HtmlRenderer(
    documentation=model.__tfi_doc__().with_updated_example_outputs(model),
    include_snapshot=model_file_fn is not None,
    extra_scripts=extra_scripts,
  )
  @app.route("/doc/en/", methods=["GET"])
  def docs():
    headers = request.headers
    return doc_renderer.render(
        proto=headers.get('X-Forwarded-Proto', 'http'),
        host=headers.get('X-Forwarded-Host', headers['HOST']),
      )

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

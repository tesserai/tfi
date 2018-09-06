import base64
import json
import inspect
import logging
import os
import os.path
import sys

import tfi.tensor.codec
from tfi.base import _recursive_transform

from collections import OrderedDict
from functools import partial
from flask import Flask, request, jsonify, send_file, send_from_directory, abort, make_response
from werkzeug.wsgi import pop_path_info, peek_path_info

from tfi import data as tfi_data
from tfi.base import _recursive_transform
from tfi.doc import documentation, render

import opentracing
from opentracing.ext import tags as opentracing_ext_tags

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
    json_data = req.get_json(force=True)
    if json_data:
        return (field in json_data, json_data.get(field, None))
    return False, None

def _default_if_empty(v, default):
    return v if v is not inspect.Parameter.empty else default

from tfi.tensor.frame import TensorFrame as _TensorFrame

# TODO(adamb) Add tracing to /specialize
# TODO(adamb) Switch _field logic to transforming to a "json" object.
#     map uploaded files to base64-encoded files. Fix tensor mapping
#     to "detect" base64 encodings for png, jpg files.

def _maybe_inject_trace_id(span, response):
    if 'Request-Id' in response.headers:
        return
    trace_id = '{:x}'.format(span.trace_id)
    response.headers['Request-Id'] = trace_id
    span.set_tag(opentracing_ext_tags.HTTP_STATUS_CODE, str(response.status_code))
    return

def _make_handler(tracer, get_span, model, method_name):
    method = getattr(model, method_name)
    sig = inspect.signature(method)
    param_annotations = {k: v.annotation for k, v in sig.parameters.items()}

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
        parent_span = get_span(request)
        if parent_span:
            parent_span.set_tag("model.method", method_name)
            parent_span.set_tag("visibility", "public")

            parent_span.log_kv({
                "http.request.header." + k.lower(): v
                for k, v in request.headers.items()
            })

            if request.form:
                parent_span.log_kv({
                    "http.request.form.%s" % k: v
                    for k, v in request.form.items(multi=True)
                })
            
            if request.data:
                parent_span.log_kv({
                    "http.request.data": request.data.decode("utf-8"),
                })

        d = {}
        for k, ann in param_annotations.items():
            ok, v = _field(request, k, ann)
            if ok:
                d[k] = v
        result = method(**d)

        r = _recursive_transform(result, _transform_value)
        if r is not None:
            result = r
        
        response = make_response(jsonify(result), 200)

        if parent_span:
            _maybe_inject_trace_id(parent_span, response)

            parent_span.log_kv({
                "http.response.header." + k.lower(): v
                for k, v in response.headers.items()
            })
            parent_span.log_kv({
                "http.response.data": response.data.decode("utf-8"),
            })

        return response
    return fn

def make_app(model,
        model_file_fn,
        operation_name_fn=None,
        extra_scripts="",
        jaeger_host=None,
        jaeger_service_name=None,
        jaeger_tags=None):
    if model is None:
        raise Exception("No model given")

    static_folder = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'static'))

    app = Flask(__name__,
            static_url_path="/static",
            static_folder=static_folder)
    _setup_logger(app, logging.DEBUG)
    tracer, get_span = _maybe_trace_app(app,
            jaeger_host=jaeger_host,
            jaeger_tags=jaeger_tags,
            operation_name_fn=operation_name_fn,
            jaeger_service_name=jaeger_service_name)

    for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
        if method_name.startswith('_'):
            continue

        fn = _make_handler(tracer, get_span, model, method_name)
        fn.__name__ = method_name
        # print("Registering", "/%s" % method_name)
        app.route("/api/%s" % method_name, methods=["POST", "GET"])(fn)

    @app.route("/meta/snapshot", methods=["GET"])
    def meta_snapshot():
        # For now we assume that this is a read-only model, so
        # just return the codepath directly.
        return send_file(model_file_fn())

    @app.route("/object/<path:objectpath>", methods=["GET"])
    def get_object(objectpath):
        return objectpath

    @app.route("/ok", methods=["GET"])
    def ok():
        return """{"status":"OK"}"""

    @app.route("/", methods=["GET"])
    def docs():
        doc_dict = documentation(model)
        return render(**doc_dict,
                      proto=request.headers.get('X-Forwarded-Proto', 'http'),
                      host=request.headers.get('X-Forwarded-Host', request.headers['HOST']),
                      extra_scripts=extra_scripts)

    return app

class make_deferred_app(object):
    def __init__(self,
            load_model_from_path_fn,
            extra_scripts="",
            jaeger_host=None,
            jaeger_service_name=None,
            jaeger_tags=None):
        # A default, empty model_app
        self._model_app = Flask(__name__)
        self._is_specialized = False

        specialize_app = Flask(__name__)
        _setup_logger(specialize_app, logging.DEBUG)
        self._specialize_app = specialize_app

        @specialize_app.route('/specialize', methods=['POST'])
        def specialize():
            codepath = '/userfunc/user'
            self._model_app = make_app(
                    load_model_from_path_fn(codepath),
                    model_file_fn=lambda: codepath,
                    jaeger_host=jaeger_host,
                    jaeger_service_name=jaeger_service_name,
                    jaeger_tags=jaeger_tags,
                    operation_name_fn=operation_name_fn,
                    extra_scripts=extra_scripts)
            self._is_specialized = True
            return ""

        @specialize_app.route('/v2/specialize', methods=['POST'])
        def v2_specialize():
            tags = dict(jaeger_tags)

            load_payload = request.get_json()
            function_metadata = load_payload.get('FunctionMetadata', {})
            print("function_metadata", function_metadata)

            function_labels = function_metadata.get('labels', {})
            account = function_labels.get('ts2-account', None)
            if account:
                tags['account'] = account

            model_name = function_metadata.get('name', None)
            if model_name:
                tags['ancestors.model'] = model_name

            codepath = load_payload['filepath']
            self._model_app = make_app(
                    load_model_from_path_fn(codepath),
                    model_file_fn=lambda: codepath,
                    jaeger_host=jaeger_host,
                    jaeger_service_name=jaeger_service_name or account or model_name,
                    operation_name_fn=lambda request, parsed_url: "HTTP %s %s %s" % (model_name, request.method, parsed_url.path),
                    jaeger_tags=tags,
                    extra_scripts=extra_scripts)
            self._is_specialized = True
            return ""

    def __call__(self, environ, start_response):
        if not self._is_specialized:
            return self._specialize_app(environ, start_response)

        if 'HTTP_X_FISSION_PARAMS_PATH_INFO' in environ:
            environ['PATH_INFO'] = '/' + environ['HTTP_X_FISSION_PARAMS_PATH_INFO']
        elif 'HTTP_X_FISSION_PARAMS_METHOD' in environ:
            environ['PATH_INFO'] = '/api/' + environ['HTTP_X_FISSION_PARAMS_METHOD']

        return self._model_app(environ, start_response)

#
# Logging setup.  TODO: Loglevel hard-coded for now. We could allow
# functions/routes to override this somehow; or we could create
# separate dev vs. prod environments.
#
def _setup_logger(app, loglevel):
    # root = logging.getLogger()
    # root.setLevel(loglevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app.logger.addHandler(ch)

def _maybe_trace_app(app, operation_name_fn=None, jaeger_host=None, jaeger_tags=None, jaeger_service_name=None):
    import opentracing

    if jaeger_host is None:
        return opentracing.Tracer(), lambda *a: None

    import jaeger_client

    # Create configuration object with enabled logging and sampling of all requests.
    config = jaeger_client.Config(config={'sampler': {'type': 'const', 'param': 1},
                            'logging': True,
                            'local_agent':
                            # Also, provide a hostname of Jaeger instance to send traces to.
                            {'reporting_host': jaeger_host}},

                    # Service name can be arbitrary string describing this particular web service.
                    service_name=jaeger_service_name or 'tfi')

    if operation_name_fn is None:
        operation_name_fn = lambda request, parsed_url: "HTTP %s %s" % (request.method, parsed_url.path)

    current_spans = {}
    traced_attributes = []
    tracer = config.initialize_tracer()
    if tracer is None:
        return opentracing.Tracer(), lambda *a: None

    from flask import _request_ctx_stack as stack
    from urllib.parse import urlparse

    def get_span(request=None):
        '''
        Returns the span tracing `request`, or the current request if
        `request==None`.

        If there is no such span, get_span returns None.

        @param request the request to get the span from
        '''
        if request is None and stack.top:
            request = stack.top.request
        # the pop call can fail if the request is interrupted by a `before_request` method so we need a default
        return current_spans.get(request, None)

    @app.before_request
    def _start_trace():
        request = stack.top.request
        parsed_url = urlparse(request.url)
        if 'X-Forwarded-Proto' in request.headers:
            parsed_url = parsed_url._replace(scheme=request.headers['X-Forwarded-Proto'])

        if 'X-Forwarded-Host' in request.headers:
            parsed_url = parsed_url._replace(netloc=request.headers['X-Forwarded-Host'])

        operation_name = operation_name_fn(request, parsed_url)
        headers = {}
        for k,v in request.headers:
            headers[k.lower()] = v

        span = None
        try:
            span_ctx = tracer.extract(opentracing.Format.HTTP_HEADERS, headers)
            span = tracer.start_span(operation_name=operation_name, child_of=span_ctx)
        except (opentracing.InvalidCarrierException, opentracing.SpanContextCorruptedException) as e:
            span = tracer.start_span(operation_name=operation_name, tags={"Extract failed": str(e)})
        if span is None:
            span = tracer.start_span(operation_name)

        span.set_tag(opentracing_ext_tags.HTTP_URL, parsed_url.geturl())
        span.set_tag(opentracing_ext_tags.HTTP_METHOD, request.method)

        if jaeger_tags:
            for k, v in jaeger_tags.items():
                span.set_tag(k, v)

        current_spans[request] = span
        for attr in traced_attributes:
            if hasattr(request, attr):
                payload = str(getattr(request, attr))
                if payload:
                    span.set_tag(attr, payload)
        
    @app.after_request
    def _after_request(response):
        span = get_span()
        if span:
            _maybe_inject_trace_id(span, response)
        return response

    @app.errorhandler(500)
    def _internal_error(error):
        response = make_response((b"Internal Error", 500))

        # TODO(adamb) Should include a log of the error
        span = get_span()
        if span:
            _maybe_inject_trace_id(span, response)

        return response

    @app.teardown_request
    def _teardown_request(exc):
        span = get_span()
        if span:
            span.finish()

    return tracer, get_span

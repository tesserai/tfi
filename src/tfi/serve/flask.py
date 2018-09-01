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
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from werkzeug.wsgi import pop_path_info, peek_path_info

from tfi import data as tfi_data
from tfi.base import _recursive_transform
from tfi.doc import documentation, render

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
    return False, None

def _default_if_empty(v, default):
    return v if v is not inspect.Parameter.empty else default

from tfi.tensor.frame import TensorFrame as _TensorFrame

def _make_handler(model, method_name):
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
        d = {}
        for k, ann in param_annotations.items():
            ok, v = _field(request, k, ann)
            if ok:
                d[k] = v
        print("args", d)
        result = method(**d)

        r = _recursive_transform(result, _transform_value)
        if r is not None:
            result = r

        return jsonify(result)
    return fn

def make_app(model, model_file_fn, extra_scripts=""):
    if model is None:
        raise Exception("No model given")

    static_folder = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'static'))

    app = Flask(__name__,
            static_url_path="/static",
            static_folder=static_folder)
    _setup_logger(app, logging.DEBUG)

    for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
        if method_name.startswith('_'):
            continue

        fn = _make_handler(model, method_name)
        fn.__name__ = method_name
        print("Registering", "/%s" % method_name)
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
    def __init__(self, load_model_from_path_fn, extra_scripts=""):
        # A default, empty model_app
        self._model_app = Flask(__name__)
        self._is_specialized = False

        codepath = '/userfunc/user'
        specialize_app = Flask(__name__)
        _setup_logger(specialize_app, logging.DEBUG)
        self._specialize_app = specialize_app

        @specialize_app.route('/specialize', methods=['POST'])
        def specialize():
            self._model_app = make_app(
                    load_model_from_path_fn(codepath),
                    model_file_fn=lambda: codepath,
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
    root = logging.getLogger()
    root.setLevel(loglevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app.logger.addHandler(ch)

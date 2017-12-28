import json
import inspect
import logging
import sys

from collections import OrderedDict
from functools import partial
from flask import Flask, request, jsonify

from tfi import data as tfi_data
from tfi import pytorch as tfi_pytorch

import tfi.tensor.codec

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

from tfi.base import _recursive_transform

import base64
def _make_handler(model, method_name):
    method = getattr(model, method_name)
    sig = inspect.signature(method)
    param_annotations = {k: v.annotation for k, v in sig.parameters.items()}
    def fn():
        d = {}
        for k, ann in param_annotations.items():
            ok, v = _field(request, k, ann)
            if ok:
                d[k] = v
        print("args", d)
        result = method(**d)

        accept_mimetypes = {
            # "image/png": lambda x: base64.b64encode(x),
            "image/png": lambda x: x,
            "text/plain": lambda x: x,
            # Use python/jsonable so we to a recursive transform before jsonification.
            "python/jsonable": lambda x: x,
        }
        r = _recursive_transform(result, lambda o: tfi.tensor.codec.encode(accept_mimetypes, o))
        if r is not None:
            result = r

        return jsonify(result)
    return fn

from tfi.doc import documentation, render

def make_app(model, extra_scripts=""):
    if model is None:
        raise Exception("No model given")

    app = Flask(__name__)
    _setup_logger(app, logging.DEBUG)

    for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
        if method_name.startswith('_'):
            continue

        fn = _make_handler(model, method_name)
        fn.__name__ = method_name
        print("Registering", "/%s" % method_name)
        app.route("/%s" % method_name, methods=["POST", "GET"])(fn)

    @app.route("/", methods=["GET"])
    def docs():
        return render(**documentation(model),
                      host=request.headers['HOST'],
                      extra_scripts=extra_scripts)

    return app

import os
def make_deferred_app(extra_scripts=""):
    app = Flask(__name__)
    _setup_logger(app, logging.DEBUG)

    model = [None]
    @app.route('/specialize', methods=['POST'])
    def load():
        codepath = '/userfunc/user'

        # load source from destination python file
        model[0] = tfi_pytorch.as_class(codepath)
        return ""

    @app.route('/', methods=['GET', 'POST', 'PUT', 'HEAD', 'OPTIONS', 'DELETE'])
    def f():
        if model[0] == None:
            print("Generic container: no requests supported")
            msg = "---HEADERS---\n%s\n--BODY--\n%s\n-----\n" % (request.headers, request.get_data())
            return msg
            # abort(500)

        print("---HEADERS---\n%s\n" % (request.headers))
        if 'X-Fission-Params-Method' not in request.headers:
            return render(
                    **documentation(model[0]),
                    host=request.headers.get('Host', "$HOST"),
                    extra_scripts=extra_scripts)

        method_name = request.headers.get('X-Fission-Params-Method', request.headers['X-Fission-Function-Name'])
        return _make_handler(model[0], method_name)()
    return app

def run_deferred(host, port, **kw):
    app = make_deferred_app(**kw)
    app.run(host=host, port=port)

def run(model, host, port, **kw):
    app = make_app(model, **kw)
    app.run(host=host, port=port)

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

import inspect
import logging
import sys

from collections import OrderedDict
from functools import partial
from flask import Flask, request, jsonify

from tfi import data as tfi_data
from tfi import pytorch as tfi_pytorch

def _field(req, field, default, annotation):
    if field in req.files:
        file = req.files[field]
        return tfi_data.file(file, mimetype=file.mimetype)
    if field in req.form:
        v = req.form[field]
        # TODO(adamb) Better to use some kind of "from_string" entry
        if 'dtype' in annotation:
            v = annotation['dtype'](v)
        return v
    return default

def _make_handler(model, method_name):
    method = getattr(model, method_name)
    sig = inspect.signature(method)
    param_annotations = {k: v.annotation for k, v in sig.parameters.items()}
    def fn():
        d = {
            k: _field(request, k, None, ann)
            for k, ann in param_annotations.items()
        }
        result = method(**d)
        return jsonify(result)
    return fn

from tfi.doc import render as render_documentation

def make_app(model):
    if model is None:
        raise Exception("No model given")

    app = Flask(__name__)
    _setup_logger(app, logging.DEBUG)

    for method_name, method in inspect.getmembers(model, predicate=inspect.ismethod):
        if method_name.startswith('_'):
            continue

        fn = _make_handler(model, method_name)
        fn.__name__ = method_name
        print("Registering", method_name)
        app.route("/%s" % method_name, methods=["POST", "GET"])(fn)

    @app.route("/", methods=["GET"])
    def docs():
        return render_documentation(model)

    return app

import os
def make_deferred_app():
    app = Flask(__name__)
    _setup_logger(app, logging.DEBUG)

    model = [None]
    @app.route('/specialize', methods=['POST'])
    def load():
        codepath = '/userfunc/user'

        # load source from destination python file
        model[0] = tfi_pytorch.as_class(codepath)
        return ""

    @app.route('/v2/specialize', methods=['POST'])
    def loadv2():
        body = request.get_json()
        filepath = body['filepath']
        model[0] = tfi_pytorch.as_class(filepath)
        return ""

    @app.route('/', methods=['GET', 'POST', 'PUT', 'HEAD', 'OPTIONS', 'DELETE'])
    def f():
        if model[0] == None:
            print("Generic container: no requests supported")
            msg = "---HEADERS---\n%s\n--BODY--\n%s\n-----\n" % (request.headers, request.get_data())
            return msg
            # abort(500)

        print("---HEADERS---\n%s\n" % (request.headers))
        if request.headers['Host'].startswith(request.headers['X-Fission-Function-Name'] + "."):
            return render_documentation(model[0])

        method_name = request.headers.get('X-Fission-Params-method', request.headers['X-Fission-Function-Name'])
        return _make_handler(model[0], method_name)()
    return app

def run_deferred(host, port):
    app = make_deferred_app()
    app.run(host=host, port=port)

def run(model, host, port):
    app = make_app(model)
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

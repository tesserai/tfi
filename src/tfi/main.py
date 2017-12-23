#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import importlib
import inspect
import os
import sys
# import tensorflow as tf
import tfi
import tfi.pytorch


from collections import OrderedDict
from functools import partial

def parse_python_str_constants(source):
    constants = {}
    parsed = ast.parse(source)
    nv = ast.NodeVisitor()
    nv.visit_Module = lambda n: nv.generic_visit(n)
    def visit_Assign(n):
        if len(n.targets) != 1:
            return

        target = n.targets[0]
        if not isinstance(target, ast.Name):
            return

        if not isinstance(n.value, ast.Str):
            return

        constants[target.id] = n.value.s

    nv.visit_Assign = visit_Assign
    nv.visit(parsed)
    return constants

def resolve_method_and_needed(obj, method_spec):
    kwargs = {}

    method_name, *rest = method_spec.split("(", 1)
    method = getattr(obj, method_name)
    if rest:
        kwargs = eval("dict(%s" % rest[0])

    sig = inspect.signature(method)
    needed = OrderedDict(sig.parameters.items())
    result = method
    # Only allow unspecified values to be given.
    for k in kwargs.keys():
        del needed[k]
    result = partial(method, **kwargs)

    return result, needed

def argparser_for_fn(fn_name, needed_params, argparse_options_fn):
    empty = inspect.Parameter.empty
    parse = argparse.ArgumentParser(prog=fn_name)
    for name, param in needed_params.items():
        parse.add_argument(
                '--%s' % name,
                required=param.default is empty,
                default=None if param.default is empty else param.default,
                **argparse_options_fn(param))

    return parse

class ModelSpecifier(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 **kwargs):
        super(ModelSpecifier, self).__init__(
            option_strings=option_strings,
            dest=dest,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
            return

        module = None
        classname = None
        rest = None
        source = None
        via_python = False
        init = {}
        if values:
            leading_value, *rest = values
        else:
            leading_value = None
            rest = []

        if leading_value is None:
            source = ""
            loaded = None
        elif leading_value.startswith('@'):
            source = leading_value[1:]
            loaded = tfi.pytorch.as_class(source)
            # loaded = tfi.saved_model.as_class(leading_value[1:])
        elif leading_value.startswith('http:') or leading_value.startswith('https:'):
            # Load exported model via http(s)
            source = leading_value
            loaded = tfi.pytorch.as_class(source)
        elif '.py:' in leading_value:
            import imp
            pre_initargs, *init_rest = leading_value.split("(", 1)
            source, classname = pre_initargs.split('.py:', 1)
            module_name = source.replace('/', '.')
            source = source + ".py"

            with open(source) as f:
                constants = parse_python_str_constants(f.read())
                if '__tfi_conda_environment_yaml__' in constants:
                    print("Found __tfi_conda_environment_yaml__.")

            module = imp.load_source(module_name, source)
            via_python = True
        else:
            # Expecting leading_value to be something like "module.class(kwargs)"
            pre_initargs, *init_rest = leading_value.split("(", 1)
            *module_fragments, classname = pre_initargs.split(".")
            module_name = ".".join(module_fragments)
            module = importlib.import_module(module_name)
            via_python = True

        if module is not None and classname is not None:
            loaded = getattr(module, classname)
            if init_rest:
                init = eval("dict(%s" % init_rest[0])

        if callable(loaded):
            sig = inspect.signature(loaded)
            needed_params = OrderedDict(sig.parameters.items())
            result = loaded
            # Only allow unspecified leading_value to be given.
            for k in init.keys():
                del needed_params[k]
            result = partial(loaded, **init)
        else:
            result = lambda: loaded
            needed_params = OrderedDict()

        p = argparser_for_fn(
                leading_value,
                needed_params,
                lambda param: {'type': type(param.default) if param.annotation is inspect.Parameter.empty else param.annotation})

        p.add_argument('_rest', default=None, nargs=argparse.REMAINDER)
        raw_args, rest_raw_args = split_list(rest, '--')
        ns = p.parse_args(raw_args)
        rest_raw_args = [*ns._rest, *rest_raw_args]
        delattr(ns, '_rest')

        setattr(namespace, self.dest, result(**vars(ns)))
        setattr(namespace, "%s_source" % self.dest, source)
        setattr(namespace, "%s_via_python" % self.dest, via_python)
        setattr(namespace, "%s_raw" % self.dest, leading_value)
        setattr(namespace, "%s_rest" % self.dest, rest_raw_args)

class _HelpAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(_HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if namespace.specifier:
            setattr(namespace, 'help', True)
            return
        parser.print_help()
        parser.exit()

parser = argparse.ArgumentParser(prog='tfi', add_help=False)
parser.add_argument('--serve', default=False, action='store_true', help='Start REST API on given port')
parser.add_argument('--publish', default=False, action='store_true', help='Publish model')
parser.add_argument('--bind', type=str, default='127.0.0.1:5000', help='Set address:port to serve on')
parser.add_argument('--export', type=str, help='path to export to')
parser.add_argument('--export-doc', type=str, help='path to export doc to')
parser.add_argument('--interactive', '-i', default=None, action='store_true', help='Start interactive session')
parser.add_argument('specifier', type=str, default=None, nargs=argparse.REMAINDER, action=ModelSpecifier, help='fully qualified class name to instantiate')
# TODO(adamb) Fix help logic.
parser.add_argument('--help', '-h', dest='help', default=None, action=_HelpAction, help="Show help")


# TODO(adamb)
#     And let's add basic text --doc output.
#     Then we'll add support for training a model locally ... (which?)
#     Then we'll add support for training a model ELSEWHERE.

def split_list(l, delim):
    for ix in range(0, len(l)):
        if l[ix] == '--':
            return l[:ix], l[ix+1:]
    return l, []

def argparser_for_fn(fn_name, needed_params, argparse_options_fn):
    empty = inspect.Parameter.empty
    parse = argparse.ArgumentParser(prog=fn_name)
    for name, param in needed_params.items():
        parse.add_argument(
                '--%s' % name,
                required=param.default is empty,
                default=None if param.default is empty else param.default,
                **argparse_options_fn(param))

    return parse

def apply_fn_args(fn_name, needed_params, param_types, fn, raw_args, chain_method=False):
    p = argparser_for_fn(fn_name, needed_params, param_types)
    ns = p.parse_args(raw_args)
    kw = vars(ns)
    return fn(**kw)

def run(argns, remaining_args):
    model = None
    exporting = argns.export is not None or argns.export_doc is not None
    serving = argns.serve is not False
    publishing = argns.publish is not False
    batch = False

    if argns.specifier:
        model = argns.specifier

        if argns.specifier_rest:
            method_name, method_raw_args = argns.specifier_rest[0], argns.specifier_rest[1:]
            method, needed = resolve_method_and_needed(model, method_name)
            result = apply_fn_args(
                    method_name,
                    needed,
                    lambda param: {
                            # 'help': "%s %s" % (tf.as_dtype(param.annotation.dtype).name, tf.TensorShape(param.annotation.tensor_shape)),
                            'type': lambda o: param.annotation.get('dtype', lambda i: i)(tfi.data.file(o[1:]) if o.startswith("@") else o),
                        },
                    method,
                    method_raw_args,
                    False)

            tensor = tfi.maybe_as_tensor(result, None, None)
            accept_mimetypes = {"image/png": tfi.format.iterm2.imgcat, "text/plain": lambda x: x}
            result_val = None
            # result_val = tfi.data._encode(tensor, accept_mimetypes)
            if result_val is None:
                result_val = result
            result_str = '%r\n' % (result_val, )
            print(result_str)
            batch = True

    if serving:
        host, port = argns.bind.split(':')
        port = int(port)
        running = False
        # try:
        #     if model is None:
        #         from tfi.serve.gunicorn import run_deferred as gunicorn_run_deferred
        #         running = True
        #         gunicorn_run_deferred(host=host, port=port)
        #     else:
        #         from tfi.serve.gunicorn import run as gunicorn_run
        #         running = True
        #         gunicorn_run(model, host=host, port=port)
        # except ModuleNotFoundError:
        #     pass

        if not running:
            if model is None:
                from tfi.serve.flask import run_deferred as flask_run_deferred
                flask_run_deferred(host=host, port=port)
            else:
                from tfi.serve.flask import run as flask_run
                flask_run(model, host=host, port=port)

    if argns.interactive is None:
        argns.interactive = not batch and not exporting and not serving and not publishing

    if argns.interactive:
        from tfi.repl import run as run_repl
        run_repl(
                globals=globals(),
                locals=None,
                history_filename=os.path.expanduser('~/.tfihistory'),
                model=model)

    if argns.export_doc:
        tfi.doc.save(argns.export_doc, model)

    if argns.export:
        tfi.pytorch.export(argns.export, model)
        # tfi.saved_model.export(argns.export, model)

    if argns.publish:
        import uuid
        import tempfile
        import hashlib
        import requests
        import json

        def sha256_for_file(f, buf_size=65536):
            pos = f.tell()
            dgst = hashlib.sha256()
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                dgst.update(data)
            size = f.tell() - pos
            f.seek(pos)

            return size, dgst.hexdigest()

        environment_name = "tfi"
        namespace = "default"
        environment = {
            "namespace": namespace,
            "name": environment_name,
        }

        fission_url = "http://35.202.47.203"
        def post(rel_url, data):
            response = requests.post(
                    "%s%s" % (fission_url, rel_url),
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"})
            # print("POST", rel_url)
            # print(response, response.text)
            if response.status_code in [404, 409]:
                return response.status_code, None
            if response.status_code == 500:
                raise Exception(response.text)
            return response.status_code, response.json()

        def get(rel_url, params=None):
            response = requests.get(
                    "%s%s" % (fission_url, rel_url),
                    params=params)
            if response.status_code == 404:
                return response.status_code, None
            if response.status_code == 500:
                raise Exception(response.text)
            return response.status_code, response.json()

        import decimal
        def format_bytes(count):
            label_ix = 0
            labels = ["B", "KiB", "MiB", "GiB"]
            while label_ix < len(labels) and count / 1024. > 1:
                count = count / 1024.
                label_ix += 1
            count = decimal.Decimal(count)
            count = count.to_integral() if count == count.to_integral() else round(count.normalize(), 2)
            return "%s %s" % (count, labels[label_ix])

        def lazily_define_package(file):
            filesize, archive_sha256 = sha256_for_file(file)
            base_archive_url = "%s/proxy/storage/v1/archive" % fission_url

            status_code, response = get("/v2/packages/%s" % archive_sha256)
            if status_code == 200:
                print("Already uploaded", flush=True)
                return archive_sha256, response

            print("Uploading %s..." % format_bytes(filesize), end='', flush=True)
            archive_response = requests.post(base_archive_url,
                    files={'uploadfile': file},
                    headers={"X-File-Size": str(filesize)})
            archive_id = archive_response.json()['id']
            print(" done", flush=True)

            archive_url = "%s?id=%s" % (base_archive_url, archive_id)

            package = {
                "metadata": {
                    "name": archive_sha256,
                    "namespace": namespace,
                },
                "spec": {
                    "environment": environment,
                    "deployment": {
                        "type": "url",
                        "url": archive_url,
                        "checksum": {
                                "type": "sha256",
                                "sum": archive_sha256,
                        },
                    },
                },
                "status": {
                    "buildstatus": "succeeded",
                },
            }
            return archive_sha256, post("/v2/packages", package)[1]

        def lazily_define_function(f):
            archive_sha256, package_ref = lazily_define_package(f)
            print("Registering ...", end='', flush=True)
            function_name = archive_sha256[:8]
            status_code, response = get("/v2/functions/%s" % function_name)
            if status_code == 200:
                return function_name

            status_code, r = post("/v2/functions", {
                "metadata": {
                    "name": function_name,
                    "namespace": namespace,
                },
                "spec": {
                    "environment": environment,
                    "package": {
                        "functionName": function_name,
                        "packageref": package_ref,
                    },
                },
            })
            if status_code == 409 or status_code == 201:
                print(" done", flush=True)
                return function_name

            print(" error", flush=True)
            raise Exception(r.text)

        def lazily_define_trigger2(function_name, http_method, host, relativeurl):
            trigger_name = "%s-%s-%s" % (
                    host.replace('.', '-'),
                    relativeurl.replace('{', '').replace('}', '').replace('/', '-'),
                    http_method.lower())
            status_code, response = get("/v2/triggers/http/%s" % trigger_name)
            if status_code == 200:
                return

            status_code, r = post("/v2/triggers/http", {
                "metadata": {
                    "name":      trigger_name,
                    "namespace": namespace,
                },
                "spec": {
                    "host": host,
                    "relativeurl": relativeurl,
                    "method":      http_method,
                    "functionref": {
                        "Type": "name",
                        "Name": function_name,
                    },
                },
            })
            if status_code == 409 or status_code == 201:
                return
            raise Exception(r.text)

        def lazily_define_trigger(f):
            function_name = lazily_define_function(f)
            host = "%s.tfi.gcp.tesserai.com" % function_name
            lazily_define_trigger2(function_name, "POST", host, "/{method}")
            lazily_define_trigger2(function_name, "GET", host, "/{method}")
            lazily_define_trigger2(function_name, "GET", host, "/")
            return "http://%s" % host

        if argns.specifier_source and not argns.specifier_via_python:
            with open(argns.specifier_source, 'rb') as f:
                url = lazily_define_trigger(f)
        else:
            with tempfile.NamedTemporaryFile(mode='rb') as f:
                print("Exporting ...", end='', flush=True)
                tfi.pytorch.export(f.name, model)
                print(" done", flush=True)
                url = lazily_define_trigger(f)
        print(url)

if __name__ == '__main__':
    run(*parser.parse_known_args(sys.argv[1:]))

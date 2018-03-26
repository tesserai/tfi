#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import importlib
import inspect
import os
import os.path
import sys
import tempfile

import tfi

from tfi.resolve.model import _detect_model_file_kind, _model_module_for_kind, _load_model_from_path_fn
from tfi.cli import resolve as _resolve_model
from tfi.tensor.codec import encode as _tfi_tensor_codec_encode
from tfi.format.iterm2 import imgcat as _tfi_format_iterm2_imgcat

from collections import OrderedDict
from functools import partial
from itertools import takewhile

def _detect_model_object_kind(model):
    klass = model if isinstance(model, type) else type(model)
    for c in klass.mro():
        if c.__name__ != "Base":
            continue
        if c.__module__ == "tfi.pytorch":
            return "pytorch"
        if c.__module__ == "tfi.tf":
            return "tensorflow"
    raise Exception("Unknown model type %s" % klass)

def _model_export(path, model):
    kind = _detect_model_object_kind(model)
    mod = _model_module_for_kind(kind)
    return mod.export(path, model)

def _model_publish(f):
    from tfi.publish import publish as _publish
    kind = _detect_model_file_kind(f)
    _publish(kind, f)

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

        if values:
            leading_value, *rest = values
        else:
            leading_value = None
            rest = []

        resolution = _resolve_model(leading_value, rest)

        setattr(namespace, self.dest, resolution['model'])
        setattr(namespace, "%s_module_fn" % self.dest, resolution.get('module_fn', lambda x: None))
        setattr(namespace, "%s_can_refresh" % self.dest, resolution.get('can_refresh', None))
        setattr(namespace, "%s_refresh_fn" % self.dest, resolution.get('refresh_fn', None))
        setattr(namespace, "%s_method_fn" % self.dest, resolution['model_method_fn'])
        setattr(namespace, "%s_source" % self.dest, resolution.get('source', None))
        setattr(namespace, "%s_source_sha1hex" % self.dest, resolution.get('source_sha1hex', None))
        setattr(namespace, "%s_via_python" % self.dest, resolution.get('via_python', None))
        setattr(namespace, "%s_raw" % self.dest, resolution.get('leading_value', None))

parser = argparse.ArgumentParser(prog='tfi', add_help=False)
parser.add_argument('--serve', default=False, action='store_true', help='Start REST API on given port')
parser.add_argument('--internal-config', type=str, default=os.environ.get("TFI_INTERNAL_CONFIG", ""), help='For internal use.')
parser.add_argument('--publish', default=False, action='store_true', help='Publish model')
parser.add_argument('--bind', type=str, default='127.0.0.1:5000', help='Set address:port to serve model on')
parser.add_argument('--export', type=str, help='path to export to')
parser.add_argument('--export-doc', type=str, help='path to export doc to')
parser.add_argument('--watch', default=False, action='store_true', help='Watch given model and reload when it changes')
parser.add_argument('--interactive', '-i', default=None, action='store_true', help='Start interactive session')
parser.add_argument('--tf-tensorboard-bind-default', type=str, default='127.0.0.1:6007')
parser.add_argument('--tf-tensorboard-bind', type=str, help='Set address:port to serve TensorBoard on. Default behavior is 127.0.0.1:6007 if available, otherwise 127.0.0.1:0')
parser.add_argument('--tf-logdir',
        default=os.path.expanduser('~/.tfi/tf/log/%F_%H-%M-%S/%04i'),
        help='Set TensorFlow log dir to write to. Renders any % placeholders with strftime, runs TensorBoard from parent dir. %04i is replaced by a 0-padded run_id count')
parser.add_argument('specifier', type=str, default=None, nargs=argparse.REMAINDER, action=ModelSpecifier, help='fully qualified class name to instantiate')


# TODO(adamb)
#     And let's add basic text --doc output.
#     Then we'll add support for training a model locally ... (which?)
#     Then we'll add support for training a model ELSEWHERE.


def run(argns, remaining_args):
    model = None
    module = None
    exporting = argns.export is not None or argns.export_doc is not None
    serving = argns.serve is not False
    publishing = argns.publish is not False
    batch = False
    if argns.interactive is None:
        argns.interactive = not batch and not exporting and not serving and not publishing

    def tf_make_logdir_fn(datetime):
        import re
        base_logdir = datetime.strftime(argns.tf_logdir)

        def logdir_fn(run_id=None):
            if run_id is None:
                return re.sub('(%\d*)i', '', base_logdir)

            base_logdir_formatstr = re.sub('(%\d*)i', '\\1d', base_logdir)
            return base_logdir_formatstr % run_id
        return logdir_fn
    import tfi
    tfi.tf_make_logdir_fn = tf_make_logdir_fn

    if argns.specifier:
        model = argns.specifier
        module = argns.specifier_module_fn()

        if argns.specifier_method_fn:
            result = argns.specifier_method_fn()

            accept_mimetypes = {"image/png": _tfi_format_iterm2_imgcat, "text/plain": lambda x: x}
            result_val = _tfi_tensor_codec_encode(accept_mimetypes, result)
            if result_val is None:
                result_val = result
            result_str = '%r\n' % (result_val, )
            print(result_str)
            batch = True

    internal_config = argns.internal_config or (model and _detect_model_object_kind(model))

    tensorboard = internal_config == 'tensorflow' and argns.interactive
    if tensorboard:
        import tfi.tf.tensorboard_server
        import threading
        tb_logdir = argns.tf_logdir
        while '%' in tb_logdir:
            tb_logdir = os.path.dirname(tb_logdir)

        if argns.tf_tensorboard_bind:
            tb_host, tb_port = argns.tf_tensorboard_bind.split(':', 1)
            tb_port = int(tb_port)
        else:
            tb_host, tb_port = argns.tf_tensorboard_bind_default.split(':', 1)
            tb_port = int(tb_port)

            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((tb_host, tb_port))
                except socket.error as e:
                    if e.errno == 98:
                        tb_port = 0

        # Use some fancy footwork to delay continuing until TensorBoard has started.
        tb_cv = threading.Condition()
        def tb_run():
            def on_ready_fn(url):
                if url:
                    print('TensorBoard at %s now serving %s' % (url, tb_logdir))
                    sys.stdout.flush()
                with tb_cv:
                    tb_cv.notify_all()
            tfi.tf.tensorboard_server.main(tb_logdir, tb_host=tb_host, tb_port=tb_port, tb_on_ready_fn=on_ready_fn)

        with tb_cv:
            tb_thread = threading.Thread(target=tb_run, daemon=True)
            tb_thread.start()
            tb_cv.wait()

    if serving:
        segment_js = """
<script>
  !function(){var analytics=window.analytics=window.analytics||[];if(!analytics.initialize)if(analytics.invoked)window.console&&console.error&&console.error("Segment snippet included twice.");else{analytics.invoked=!0;analytics.methods=["trackSubmit","trackClick","trackLink","trackForm","pageview","identify","reset","group","track","ready","alias","debug","page","once","off","on"];analytics.factory=function(t){return function(){var e=Array.prototype.slice.call(arguments);e.unshift(t);analytics.push(e);return analytics}};for(var t=0;t<analytics.methods.length;t++){var e=analytics.methods[t];analytics[e]=analytics.factory(e)}analytics.load=function(t){var e=document.createElement("script");e.type="text/javascript";e.async=!0;e.src=("https:"===document.location.protocol?"https://":"http://")+"cdn.segment.com/analytics.js/v1/"+t+"/analytics.min.js";var n=document.getElementsByTagName("script")[0];n.parentNode.insertBefore(e,n)};analytics.SNIPPET_VERSION="4.0.0";
  analytics.load("GaappI2dkNZV4PLVdiJ8pHQ7Hofbf6Vz");
  analytics.page();
  }}();
</script>
        """

        host, port = argns.bind.split(':')
        prefork_ok = True
        port = int(port)
        if model is None:
            if internal_config == 'tensorflow':
                prefork_ok = False

            from tfi.serve import run_deferred as serve_deferred
            serve_deferred(
                    host=host, port=port,
                    prefork_ok=prefork_ok,
                    load_model_from_path_fn=_load_model_from_path_fn,
                    extra_scripts=segment_js)
        else:
            if internal_config == 'tensorflow':
                prefork_ok = False

            from tfi.serve import run as serve
            def model_file_fn():
                if argns.specifier_source and not argns.specifier_via_python:
                    return argns.specifier_source
                with tempfile.NamedTemporaryFile(mode='rb', delete=False) as f:
                    print("Exporting ...", end='', flush=True)
                    _model_export(f.name, model)
                    print(" done", flush=True)
                    return f.name
            serve(model,
                    host=host,
                    port=port,
                    prefork_ok=prefork_ok,
                    extra_scripts=segment_js,
                    model_file_fn=model_file_fn)

    if argns.watch:
        if not argns.specifier_can_refresh:
            print("WARN: Can't watch unrefreshable model.")
        else:
            import tfi.watch
            ar = tfi.watch.AutoRefresher()
            def do_refresh():
                def refresh_progress(model, ix, total):
                    print("Refreshing %d/%d: %s" % (ix, total, model))
                argns.specifier_refresh_fn(refresh_progress)
            ar.watch(argns.specifier_source, argns.specifier_source_sha1hex, do_refresh)
            ar.start()

    if argns.interactive:
        from tfi.repl import run as run_repl
        run_repl(
                globals=globals(),
                locals=None,
                history_filename=os.path.expanduser('~/.tfihistory'),
                model=model,
                module=module)

    if argns.export_doc:
        tfi.doc.save(argns.export_doc, model)

    if argns.export:
        if argns.specifier_source and not argns.specifier_via_python:
            import shutil
            shutil.copyfile(argns.specifier_source, argns.export)
        else:
            _model_export(argns.export, model)

    if argns.publish:
        if argns.specifier_source and not argns.specifier_via_python:
            with open(argns.specifier_source, 'rb') as f:
                # TODO(adamb) Should actually autodetect which environment to use.
                url = _model_publish(f)
        else:
            with tempfile.NamedTemporaryFile(mode='rb') as f:
                # TODO(adamb) Should actually autodetect which environment to use.
                print("Exporting ...", end='', flush=True)
                _model_export(f.name, model)
                print(" done", flush=True)
                url = _model_publish(f)
        print(url)

def main():
    argns, remaining_args = parser.parse_known_args(sys.argv[1:])
    argns.load_model_from_path_fn = _load_model_from_path_fn
    run(argns, remaining_args)

if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import importlib
import inspect
import os
import sys
import tensorflow as tf
import tfi


from collections import OrderedDict
from functools import partial

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

        init = {}
        if values.startswith('@'):
            klass = tfi.saved_model.as_class(values[1:])
        else:
            # Expecting values to be something like "module.class(kwargs)"
            pre_initargs, *rest = values.split("(", 1)
            *module_fragments, classname = pre_initargs.split(".")
            module_name = ".".join(module_fragments)
            module = importlib.import_module(module_name)
            klass = getattr(module, classname)
            if rest:
                init = eval("dict(%s" % rest[0])

        sig = inspect.signature(klass)
        needed = OrderedDict(sig.parameters.items())
        result = klass
        # Only allow unspecified values to be given.
        for k in init.keys():
            del needed[k]
        result = partial(klass, **init)

        setattr(namespace, self.dest, result)
        setattr(namespace, "%s_raw" % self.dest, values)
        setattr(namespace, "%s_kwargs" % self.dest, needed)

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
parser.add_argument('--export', type=str, help='path to export to')
parser.add_argument('--interactive', '-i', default=False, action='store_true', help='Start interactive session')
parser.add_argument('specifier', type=str, default=None, nargs='?', action=ModelSpecifier, help='fully qualified class name to instantiate')
parser.add_argument('method', type=str, nargs='?', help='name of method to run')
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

def apply_fn_args(fn_name, needed_params, param_types, fn, raw_args):
    p = argparser_for_fn(fn_name, needed_params, param_types)
    kw = vars(p.parse_args(raw_args))
    return fn(**kw)


def run(argns, remaining_args):
    model = None
    exporting = argns.export is not None
    if argns.specifier:
        hparam_raw_args, method_raw_args = split_list(remaining_args, '--')

        model = apply_fn_args(
                argns.specifier_raw,
                argns.specifier_kwargs,
                lambda param: {'type': type(param.default) if param.annotation is inspect.Parameter.empty else param.annotation},
                argns.specifier,
                hparam_raw_args)

        if method_raw_args:
            method_name, method_raw_args = method_raw_args[0], method_raw_args[1:]
            method = getattr(model, method_name)
            apply_fn_args(
                    method_name,
                    inspect.signature(method).parameters,
                    lambda param: {'help': "%s %s" % (tf.as_dtype(param.annotation.dtype).name, tf.TensorShape(param.annotation.tensor_shape)) },
                    method,
                    method_raw_args)
        else:
            argns.interactive = not exporting
    else:
        argns.interactive = True

    if argns.interactive:
        from tfi.repl import run as run_repl
        run_repl(
                globals=globals(),
                locals=None,
                history_filename=os.path.expanduser('~/.tfihistory'),
                model=model)

    if argns.export:
        tfi.saved_model.export(argns.export, model)

if __name__ == '__main__':
    run(*parser.parse_known_args(sys.argv[1:]))

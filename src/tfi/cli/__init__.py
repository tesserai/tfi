import argparse
import inspect

from collections import OrderedDict
from functools import partial

from tfi.base import _GetAttrAccumulator as _GetAttrAccumulator
from tfi.data import file as _tfi_data_file
from tfi.resolve.model import resolve_exported as _resolve_exported
from tfi.resolve.model import resolve_url as _resolve_url
from tfi.resolve.model import resolve_python_source as _resolve_python_source
from tfi.resolve.model import resolve_module as _resolve_module

def _split_list(l, delim):
    for ix in range(0, len(l)):
        if l[ix] == '--':
            return l[:ix], l[ix+1:]
    return l, []

def _resolve_needed_params(method, have_kwargs=None):
    sig = inspect.signature(method)
    needed = OrderedDict(sig.parameters.items())
    if inspect.isfunction(method):
        del needed[list(needed.keys())[0]]

    # Only allow unspecified values to be given.
    if have_kwargs:
        for k in have_kwargs.keys():
            del needed[k]

    return needed

def resolve(model_class_from_path_fn, leading_value, rest):
    if leading_value is None:
        resolution = {
            'source': "",
            'loaded': None,
        }
    elif leading_value.startswith('@'):
        resolution = _resolve_exported(model_class_from_path_fn, leading_value[1:])
    elif leading_value.startswith('http://') or leading_value.startswith('https://'):
        resolution = _resolve_url(model_class_from_path_fn, leading_value)
    elif '.py:' in leading_value:
        resolution = _resolve_python_source(leading_value)
    else:
        resolution = _resolve_module(leading_value)

    if 'model_fn_needed_params' not in resolution:
        resolution['model_method_fn'] = None
        resolution['model'] = None
        return resolution

    empty = inspect.Parameter.empty

    p = argparse.ArgumentParser(prog=leading_value)

    for name, param in resolution['model_fn_needed_params'].items():
        p.add_argument(
            '--%s' % name,
            required=param.default is empty,
            default=None if param.default is empty else param.default,
            type=type(param.default) if param.annotation is inspect.Parameter.empty else param.annotation,
        )
    p.set_defaults(_method=None)

    def apply_fn(ns_keys_to_kw, fn, ns):
        kw = {}
        for ns_k, kw_k in ns_keys_to_kw.items():
            if hasattr(ns, ns_k):
                kw[kw_k] = getattr(ns, ns_k)
        return fn(**kw)

    def apply_model_method(method_name, ns_keys_to_kw, model, ns):
        return apply_fn(ns_keys_to_kw, getattr(model, method_name), ns)

    def parse_arg_fn(annotation):
        dtype_fn = annotation.get('dtype', lambda i: i)
        return lambda o: dtype_fn(_tfi_data_file(o[1:]) if o.startswith("@") else o)

    subparsers = p.add_subparsers(help='sub-command help')
    for membername, member in resolution['model_members']:
        sp = subparsers.add_parser(membername)
        needed_params = _resolve_needed_params(member)
        ns_keys_to_kw = {}
        for name, param in needed_params.items():
            # HACK(adamb) Should actually properly process theses?!?
            if isinstance(param.annotation, _GetAttrAccumulator):
                continue
            dest = "_%s.%s" % (membername, name)
            ns_keys_to_kw[dest] = name
            sp.add_argument(
                '--%s' % name,
                required=param.default is empty,
                dest=dest,
                metavar=name.upper(),
                default=None if param.default is empty else param.default,
                type=parse_arg_fn(param.annotation),
            )
        sp.set_defaults(_method=partial(apply_model_method, membername, ns_keys_to_kw))

    ns = p.parse_args(rest)
    model_fn = resolution['model_fn']
    model = apply_fn(
        {k: k for k in resolution['model_fn_needed_params'].keys()},
        model_fn,
        ns)

    resolution['model_method_fn'] = partial(ns._method, model, ns) if ns._method else None
    resolution['model'] = model
    return resolution

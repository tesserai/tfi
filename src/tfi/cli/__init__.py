import ast
import argparse
import inspect

from collections import OrderedDict
from functools import partial

from tfi.base import _GetAttrAccumulator as _GetAttrAccumulator
from tfi.data import file as _tfi_data_file
from tfi.resolve.model import resolve_auto as _resolve_auto

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

def _parse_arg_fn(annotation):
    dtype_fn = None
    if isinstance(annotation, dict):
        dtype_fn = annotation.get('dtype', None)
    elif hasattr(annotation, 'dtype'):
        dtype_fn = annotation.dtype
    
    def default_dtype_fn(s):
        print("default_dtype_fn", s)
        if s:
            ch = s[0]
            if ch == '[' or ch == '{' or ch.isdecimal():
                return ast.literal_eval(s)
        return s
    default_dtype_fn.__name__ = 'literal'
    if dtype_fn is None:
        dtype_fn = default_dtype_fn
    return lambda o: dtype_fn(_tfi_data_file(o[1:]) if o.startswith("@") else o)

def resolve(leading_value, rest):
    resolution = _resolve_auto(leading_value)
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
            type=_parse_arg_fn({} if param.annotation is empty else param.annotation),
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
                type=_parse_arg_fn({} if param.annotation is empty else param.annotation ),
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

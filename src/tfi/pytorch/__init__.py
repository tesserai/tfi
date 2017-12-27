from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict, namedtuple
import inspect
import functools
import os
import re
import sys
import types
import warnings

from tfi.base import _GetAttrAccumulator
from tfi.parse.docstring import GoogleDocstring

from tfi.pytorch.tensor_codec import as_tensor

import torch

def _resolve_instance_method_tensors(instance, fn):
    def _expand_annotation(instance, annotation, default=None):
        if annotation == inspect.Signature.empty:
            return default
        return annotation(instance) if isinstance(annotation, _GetAttrAccumulator) else annotation

    def _expand_annotation_dict(instance, annotation, default=None):
        if annotation == inspect.Signature.empty:
            return default
        return {
            k: v(instance) if isinstance(v, _GetAttrAccumulator) else v
            for k, v in annotation.items()
        }

    def _tensor_info_str(tensor):
        shape_list = tensor['shape']
        ndims = len(shape_list)
        dtype_name = tensor.get('dtype', None)
        if ndims is None:
            return "%s ?" % dtype_name

        return "%s <%s>" % (
            dtype_name,
            ", ".join(["?" if n is None else str(n) for n in shape_list]),
        )

    def _enrich_docs(doc_fields, tensor_dict):
        existing = {k: v for k, _, v in doc_fields}
        return [
            (
                name,
                _tensor_info_str(tensor_dict[name]) if name in tensor_dict else '',
                existing.get(name, '')
            )
            for name in set([*tensor_dict.keys(), *existing.keys()])
        ]

    sig = inspect.signature(fn)
    input_annotations = OrderedDict([
        (name, _expand_annotation_dict(instance, param.annotation))
        for name, param in sig.parameters.items()
    ])
    output_annotations = OrderedDict([
        (name, _expand_annotation_dict(instance, value))
        for name, value in _expand_annotation(instance, sig.return_annotation, {}).items()
    ])

    if fn.__doc__:
        doc = GoogleDocstring(obj=fn).result()
    else:
        doc = {'sections': [], 'args': {}, 'returns': {}}
    doc['args'] = _enrich_docs(doc['args'], input_annotations)
    doc['returns'] = _enrich_docs(doc['returns'], output_annotations)

    return doc, input_annotations, output_annotations

def _make_method(signature_def, existing):
    input_tensor_names = signature_def['inputs'].keys()

    def session_handle_for(value, signature_def_input):
        if isinstance(value, float):
            pass
        else:
            value = as_tensor(value, signature_def_input['shape'], signature_def_input.get('dtype', None))
        if 'transform' in signature_def_input:
            value = signature_def_input['transform'](value)
        if signature_def_input.get('dtype', None) in (int,):
            return value
        return torch.autograd.Variable(value, volatile=True)

    def _impl(self, **kwargs):
        return existing(**{
            input_name: session_handle_for(kwargs[input_name], input_d)
            for input_name, input_d in signature_def['inputs'].items()
        })

    # Need to properly forge method parameters, complete with annotations.
    argdef = ",".join(["_", *input_tensor_names])
    argcall = ",".join(["_", *["%s=%s" % (k, k) for k in input_tensor_names]])
    gensrc = """lambda %s: _impl(%s)""" % (argdef, argcall)
    impl = eval(gensrc, {'_impl': _impl})
    sigdef_inputs = signature_def['inputs']
    impl.__annotations__ = {
        k: sigdef_inputs[k]
        for k, p in inspect.signature(impl).parameters.items()
        if k in sigdef_inputs
    }
    return impl

class Meta(type):
    @staticmethod
    def __new__(meta, classname, bases, d):
        if '__tfi_del__' in d:
            for name in d['__tfi_del__']:
                del d[name]
            del d['__tfi_del__']

        if '__init__' in d:
            init = d['__init__']

            # Wrap __init__ to auto adapt inputs.
            @functools.wraps(init)
            def wrapped_init(self, *a, **k):
                init(self, *a, **k)

                # Once init has executed, we can bind proper methods too!
                if not hasattr(self, '__tfi_signature_defs__'):
                    self.__tfi_signature_defs__ = OrderedDict()
                    self.__tfi_signature_def_docs__ = OrderedDict()

                    for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
                        if method_name.startswith('_'):
                            continue

                        doc, input_annotations, output_annotations = _resolve_instance_method_tensors(self, method)

                        self.__tfi_signature_def_docs__[method_name] = doc
                        self.__tfi_signature_defs__[method_name] = dict(
                            inputs=input_annotations,
                            outputs=output_annotations)

                # Remember which fields to pickle BEFORE we add methods.
                if not hasattr(self, '__getstate__'):
                    self.__tfi_saved_fields__ = list(self.__dict__.keys())
                    self.__getstate__ = lambda: {k: getattr(self, k) for k in self.__tfi_saved_fields__}

                self.__tfi_init__()
            d['__init__'] = wrapped_init

        return super(Meta, meta).__new__(meta, classname, bases, d)

    @staticmethod
    def __prepare__(name, bases):
        def input_name_decorator(name, **kwargs):
            def install_annotation(fn):
                # TODO(adamb) Should blow up if unknown/invalid kwargs are given.
                # TODO(adamb) Should blow up if kwargs are repeated.
                # TODO(adamb) Should blow up if there is no such argument
                fn.__annotations__[name] = kwargs
                return fn
            return install_annotation

        d = OrderedDict({
            'tfi_input': input_name_decorator,
            'self': _GetAttrAccumulator(),
        })
        # NOTE(adamb) Remember to delete all of these! Every item in here is
        #     essentially "reserved" and can't be used as a method name in a
        #     SavedModel. Expand it with caution.
        d['__tfi_del__'] = list(d.keys())

        return d

class Base(object, metaclass=Meta):
    def __tfi_init__(self):
        for method_name, sigdef in self.__tfi_signature_defs__.items():
            setattr(self,
                    method_name,
                    types.MethodType(
                            _make_method(sigdef, getattr(self, method_name)),
                            self))

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        self.__tfi_init__()

from tfi.pytorch.load import persistent_load
from tfi.pytorch import kosher as _kosher
from tfi.doc import record_documentation

def export(export_path, model):
    record_documentation(model)
    pickle_module = _kosher.PickleModule(lambda m: m.startswith('zoo.'))
    pickle_module.persistent_load = persistent_load
    with open(export_path, "w+b") as f:
        torch.save(model, f, pickle_module=pickle_module)

def as_class(import_path):
    with open(import_path, "rb") as f:
        return torch.load(f, pickle_module=_kosher.PickleModule(lambda x: False))

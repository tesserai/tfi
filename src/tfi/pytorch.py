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

from tfi.as_tensor import as_tensor
from tfi.doc.docstring import GoogleDocstring

import torch

class _GetAttrAccumulator:
    def __init__(self, gotten=None):
        if gotten is None:
            gotten = []
        self._gotten = gotten

    def __getattr__(self, name):
        return _GetAttrAccumulator([*self._gotten, name])

    def __call__(self, target):
        result = target
        for name in self._gotten:
            result = getattr(result, name)
        return result


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
            (name, _tensor_info_str(tensor), existing.get(name, ''))
            for name, tensor in tensor_dict.items()
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
                    saved_fields = list(self.__dict__.keys())
                    self.__getstate__ = lambda: {k: getattr(self, k) for k in saved_fields}

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

deserialized_objects = {}
restore_location = torch.serialization.default_restore_location
def _check_container_source(container_type, source_file, original_source):
    current_source = inspect.getsource(container_type)
    if original_source != current_source:
        if container_type.dump_patches:
            file_name = container_type.__name__ + '.patch'
            diff = difflib.unified_diff(current_source.split('\n'),
                                        original_source.split('\n'),
                                        source_file,
                                        source_file, lineterm="")
            lines = '\n'.join(diff)
            try:
                with open(file_name, 'a+') as f:
                    file_size = f.seek(0, 2)
                    f.seek(0)
                    if file_size == 0:
                        f.write(lines)
                    elif file_size != len(lines) or f.read() != lines:
                        raise IOError
                msg = ("Saved a reverse patch to " + file_name + ". "
                       "Run `patch -p0 < " + file_name + "` to revert your "
                       "changes.")
            except IOError:
                msg = ("Tried to save a patch, but couldn't create a "
                       "writable file " + file_name + ". Make sure it "
                       "doesn't exist and your working directory is "
                       "writable.")
        else:
            msg = ("you can retrieve the original source code by "
                   "accessing the object's source attribute or set "
                   "`torch.nn.Module.dump_patches = True` and use the "
                   "patch tool to revert the changes.")
        msg = ("source code of class '{}' has changed. {}"
               .format(torch.typename(container_type), msg))
        warnings.warn(msg, SourceChangeWarning)

def _persistent_load(saved_id):
    assert isinstance(saved_id, tuple)
    typename = saved_id[0]
    data = saved_id[1:]

    if typename == 'module':
        # Ignore containers that don't have any sources saved
        if all(data[1:]):
            _check_container_source(*data)
        return data[0]
    elif typename == 'storage':
        data_type, root_key, location, size, view_metadata = data
        if root_key not in deserialized_objects:
            deserialized_objects[root_key] = restore_location(
                data_type(size), location)
        storage = deserialized_objects[root_key]
        if view_metadata is not None:
            view_key, offset, view_size = view_metadata
            if view_key not in deserialized_objects:
                deserialized_objects[view_key] = storage[offset:offset + view_size]
            return deserialized_objects[view_key]
        else:
            return storage
    else:
        raise RuntimeError("Unknown saved id type: %s" % saved_id[0])

import tfi.kosher
def export(export_path, model):
    pickle_module = tfi.kosher.PickleModule(lambda m: m.startswith('zoo.'))
    pickle_module.persistent_load = _persistent_load
    with open(export_path, "w+b") as f:
        torch.save(model, f, pickle_module=pickle_module)

def as_class(import_path):
    with open(import_path, "rb") as f:
        return torch.load(f, pickle_module=tfi.kosher.PickleModule(lambda x: False))

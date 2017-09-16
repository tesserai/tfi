from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict, namedtuple
import inspect
import os
import re
import sys
import types
import warnings

import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader

from tfi.as_tensor import as_tensor

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

def _signature_def_for_instance_method(instance, fn):
    def _expand_annotation(instance, annotation, default=None):
        if annotation == inspect.Signature.empty:
            return default
        return annotation(instance) if callable(annotation) else annotation

    def _get_tfi_method_name(fn):
        if hasattr(fn, '__tfi_method_name__'):
            return fn.__tfi_method_name__
        return None

    def _tensor_infos_dict(tensor_dict):
        return OrderedDict([
            (name, tf.saved_model.utils.build_tensor_info(tensor))
            for name, tensor in tensor_dict.items()
        ])

    sig = inspect.signature(fn)
    input_tensors = OrderedDict([
        (name, _expand_annotation(instance, param.annotation))
        for name, param in sig.parameters.items()
    ])
    output_tensors = OrderedDict([
        (name, _expand_annotation(instance, value))
        for name, value in _expand_annotation(instance, sig.return_annotation, {}).items()
    ])
    return tf.saved_model.signature_def_utils.build_signature_def(
        inputs=_tensor_infos_dict(input_tensors),
        outputs=_tensor_infos_dict(output_tensors),
        method_name=_get_tfi_method_name(fn))

def _make_method(signature_def, result_class_name):
    # Sort to preserve order because we need to go from value to key later.
    output_tensor_keys_sorted = sorted(signature_def.outputs.keys())
    output_tensor_names_sorted = [
        signature_def.outputs[tensor_key].name
        for tensor_key in output_tensor_keys_sorted
    ]

    result_class = namedtuple(result_class_name, output_tensor_keys_sorted)
    input_tensor_names_sorted = sorted(signature_def.inputs.keys())

    def session_handle_for(value, signature_def_input):
        if isinstance(value, float):
            return None
        # TODO(adamb) This might append to the default graph. These appends
        #     should be cached and idempotent. The added nodes should be
        #     reused if possible.
        return as_tensor(
            value,
            tf.TensorShape(signature_def_input.tensor_shape).as_list(),
            tf.as_dtype(signature_def_input.dtype))

    def _run(self, **kwargs):
        feed_dict = {}
        session = self.__tfi_session__
        with session.graph.as_default():
            handle_keys = []
            handle_in = []
            for key in input_tensor_names_sorted:
                input = signature_def.inputs[key]
                value = kwargs[key]
                sh = session_handle_for(value, input)
                if sh is None:
                    feed_dict[input.name] = value
                else:
                    handle_keys.append(input.name)
                    handle_in.append(sh)

            for key, fh in zip(handle_keys, session.run(handle_in)):
                feed_dict[key] = fh

            result = session.run(
                output_tensor_names_sorted,
                feed_dict=feed_dict)

        return result_class._make(result)
    return _run

class Meta(type):
    @staticmethod
    def __new__(meta, classname, bases, d):
        if '__tfi_del__' in d:
            for name in d['__tfi_del__']:
                del d[name]
            del d['__tfi_del__']

        if '__init__' in d:
            init = d['__init__']
            # Wrap __init__ to graph, session, and methods.
            def wrapped_init(self, *a, **k):
                self.__tfi_graph__ = tf.Graph()
                self.__tfi_session__ = tf.Session(graph=self.__tfi_graph__)
                with self.__tfi_graph__.as_default():
                    with self.__tfi_session__.as_default():
                        init(self, *a, **k)

                # Once init has executed, we can bind proper methods too!
                if not hasattr(self, '__tfi_signature_defs__'):
                    self.__tfi_signature_defs__ = OrderedDict()
                    for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
                        if method_name.startswith('_'):
                            continue
                        self.__tfi_signature_defs__[method_name] = _signature_def_for_instance_method(self, method)

                for method_name, sigdef in self.__tfi_signature_defs__.items():
                    result_class_name = '%s__%s__Result' % (classname, method_name)
                    setattr(self,
                            method_name,
                            types.MethodType(
                                    _make_method(sigdef, result_class_name),
                                    self))

            d['__init__'] = wrapped_init

        return super(Meta, meta).__new__(meta, classname, bases, d)

    @staticmethod
    def __prepare__(name, bases):
        def method_name_decorator(name):
            def install_tfi_method_name(fn):
                fn.__tfi_method_name__ = name
                return fn
            return install_tfi_method_name

        d = OrderedDict({
            'tfi_method_name': method_name_decorator,
            'self': _GetAttrAccumulator(),
        })
        # NOTE(adamb) Remember to delete all of these! Every item in here is
        #     essentially "reserved" and can't be used as a method name in a
        #     SavedModel. Expand it with caution.
        d['__tfi_del__'] = list(d.keys())

        return d

class Base(object, metaclass=Meta):
    pass

def _get_meta_graph_def(saved_model_dir, tag_set):
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def

  raise RuntimeError('MetaGraphDef associated with tag-set ' + tag_set +
                     ' could not be found in SavedModel')

def as_class(saved_model_dir, tag_set=tf.saved_model.tag_constants.SERVING):
    # TODO(adamb) Choose a better name than CUSTOMTYPE.
    return type('CUSTOMTYPE', (Base,), {
        '__init__': lambda s:
                loader.load(tf.get_default_session(), tag_set.split(','), saved_model_dir),
        '__tfi_signature_defs__':
                _get_meta_graph_def(saved_model_dir, tag_set).signature_def,
    })

def export(export_path, model_fn):
    # TODO(adamb) Allow customization of tags.
    tags = [tf.saved_model.tag_constants.SERVING]

    model = model_fn()
    if not isinstance(model, Base):
        raise Exception("%s is not an instance of Base" % model)

    with model.__tfi_graph__.as_default():
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
              model.__tfi_session__,
              tags,
              legacy_init_op=legacy_init_op,
              signature_def_map=model.__tfi_signature_defs__)
        builder.save()

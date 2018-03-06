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

import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import ops as ops_lib

from tfi.base import _GetAttrAccumulator

from tfi.parse.docstring import GoogleDocstring

from tfi.tensor.codec import _BaseAdapter
from tfi.tf.tensor_codec import as_tensor

import tfi.tf.checkpoint

def _walk(out_tensors, in_tensors, fn):
    to_visit = set(out_tensors)
    visited = set()

    while to_visit:
        obj = to_visit.pop()

        if isinstance(obj, (tf.Tensor, tf.Variable)):
            op = obj.op
        elif isinstance(obj, tf.Operation):
            op = obj
        else:
            raise Exception("Encountered unwalkable object: %r" % obj)

        visited.add(obj)
        fn(obj)

        if obj in in_tensors:
            continue # Don't walk past in_tensors

        for dep in op.inputs:
            if dep not in visited:
                to_visit.add(dep)

        for dep in op.control_inputs:
            if dep not in visited:
                to_visit.add(dep)

from tensorflow.contrib.graph_editor import make_view
import tensorflow.contrib.graph_editor.transform as transform

class LazyObject(object):
    __slots__ = ('_attrs', '_resolve_fn', '_resolving', '_frozen')
    def __init__(self, resolve_fn):
        self._attrs = {}
        self._resolve_fn = resolve_fn
        self._resolving = []
        self._frozen = True

    def __setattr__(self, name, value):
        if hasattr(self, '_frozen') and self._frozen:
            raise Exception("Can't set attribute outside constructor")
        return object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in ('_attrs', '_resolve_fn', '_resolving', '_frozen'):
            return object.__getattr__(self, name)

        if name in self._attrs:
            return self._attrs[name]
        if name in self._resolving:
            raise Exception("Encountered a cyclical dependency while resolving %s. Depchain: %s" % (name, self._resolving))
        self._resolving.append(name)
        try:
            value = self._resolve_fn(name)
        finally:
            self._resolving.pop()
        self._attrs[name] = value
        return value

def _build_signature_def(input_tensors, output_tensors, method_name):
    def _tensor_infos_dict(tensor_dict):
        return OrderedDict([
            (name, tf.saved_model.utils.build_tensor_info(tensor))
            for name, tensor in tensor_dict.items()
        ])

    return tf.saved_model.signature_def_utils.build_signature_def(
                inputs=_tensor_infos_dict(input_tensors),
                outputs=_tensor_infos_dict(output_tensors),
                method_name=method_name)

class _CopyingCall(object):
    def __init__(self, input_tensors, output_tensors):
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._sgv = None

    def __call__(self, **kwargs):
        if len(kwargs) != len(self._input_tensors):
            raise Exception("Wrong arguments. Expected: %s, got %s" % (
                    list(self._input_tensors.keys()),
                    list(kwargs)))

        replacement_ts = {
            tensor: kwargs[name]
            for name, tensor in self._input_tensors.items()
        }

        if not self._sgv:
            # NOTE(adamb) This seems to break export/import logic when applied to graphs
            #     with at least the Adam optimizer.
            method_ops = set()
            input_tensors = set(self._input_tensors.values())
            input_ops = {t.op for t in input_tensors}
            def _append_op(o):
                if isinstance(o, tf.Tensor):
                    if o in input_tensors:
                        return
                    o = o.op
                if o in input_ops:
                    return
                # Don't COPY any variables, just use them as-is!
                # NOTE(adamb) There may be scenarios where this is *not*
                #     what we want.
                if o.type == "VariableV2":
                    return
                method_ops.add(o)

            _walk(self._output_tensors.values(),
                  input_tensors,
                  _append_op)

            graph = list(self._output_tensors.values())[0].graph
            self._sgv = make_view(list(method_ops), graph=graph)

        new_sgv, tinfo = transform.copy_with_input_replacements(
            self._sgv,
            replacement_ts,
            dst_scope="")

        return {
            name: tinfo.transformed(t)
            for name, t in self._output_tensors.items()
        }

class _RenderedInstanceMethod(object):
    def __init__(self, doc, input_tensors, output_tensors, method_name, call_fn):
        self.doc = doc
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._method_name = method_name
        self._call_fn = call_fn

    def __call__(self, **kwargs):
        if len(kwargs) != len(self._input_tensors):
            raise Exception("Wrong arguments. Expected: %s, got %s" % (
                    list(self._input_tensors.keys()),
                    list(kwargs)))
        return self._call_fn(**kwargs)

    def build_signature_def(self):
        return _build_signature_def(self._input_tensors, self._output_tensors, self._method_name)

import dis
def _empty_method():
    pass
def _dump_instructions(fn):
    return [
        (isn.opcode, isn.opname, isn.argval, isn.offset, isn.is_jump_target)
        for isn in dis.get_instructions(fn)
    ]
EMPTY_METHOD_INSTRUCTIONS = _dump_instructions(_empty_method)

def is_empty_method(f):
    val = _dump_instructions(f)
    return EMPTY_METHOD_INSTRUCTIONS == val

def _resolve_instance_method_tensors(lazy_instance, method):
    def _expand_annotation(annotation, default=None):
        if annotation == inspect.Signature.empty:
            return default
        return _GetAttrAccumulator.apply(annotation, lazy_instance)

    def _tensor_info_str(tensor):
        if tensor.shape.ndims is None:
            return '%s ?' % tensor.dtype.name

        return '%s <%s>' % (
            tensor.dtype.name,
            ', '.join(['?' if n is None else str(n) for n in tensor.shape.as_list()]),
        )

    def _enrich_docs_with_tensor_info(doc_fields, tensor_dict):
        existing = {k: v for k, _, v in doc_fields}
        return [
            (name, _tensor_info_str(tensor), existing.get(name, ''))
            for name, tensor in tensor_dict.items()
        ]

    def _enriched_method_doc(input_tensors, output_tensors):
        if method.__doc__:
            doc = GoogleDocstring(obj=method).result()
        else:
            doc = {'sections': [], 'args': {}, 'returns': {}}
        doc['args'] = _enrich_docs_with_tensor_info(doc['args'], input_tensors)
        doc['returns'] = _enrich_docs_with_tensor_info(doc['returns'], output_tensors)
        return doc

    sig = inspect.signature(method)

    input_annotations = OrderedDict([
        (name, _expand_annotation(param.annotation))
        for name, param in sig.parameters.items()
    ])

    if is_empty_method(method):
        input_tensors = input_annotations
        output_tensors = _expand_annotation(sig.return_annotation, default={})
        call_fn = _CopyingCall(input_tensors, output_tensors)
    else:
        # If method isn't empty, assume annotations on parameters are kwargs for tf.placeholder
        # Make the resulting graph a bit nicer to read with scope names:
        # <method>/inputs, <method>/_, <method>/outputs.
        with tf.name_scope(method.__name__):
            with tf.name_scope("inputs"):
                input_tensors = {
                    name: tf.placeholder(name=name, **placeholder_kwargs)
                    for name, placeholder_kwargs in input_annotations.items()
                }
            with tf.name_scope("_"):
                output_tensors = method(**input_tensors)
            with tf.name_scope("outputs"):
                output_tensors = {
                    output_name: tf.identity(t, name=output_name)
                    for output_name, t in output_tensors.items()
                }
        call_fn = method

    if not output_tensors:
        raise Exception("No output tensors for %s" % method)

    return _RenderedInstanceMethod(
            doc=_enriched_method_doc(input_tensors, output_tensors),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            method_name=method.__tfi_method_name__ if hasattr(method, '__tfi_method_name__') else None,
            call_fn=call_fn)

def _make_method(instance, signature_def, var_list):
    # print("_make_method", signature_def)
    # Sort to preserve order because we need to go from value to key later.
    output_tensor_keys_sorted = sorted(signature_def.outputs.keys())
    output_tensor_names_sorted = [
        signature_def.outputs[tensor_key].name
        for tensor_key in output_tensor_keys_sorted
    ]

    input_tensor_names_sorted = sorted(signature_def.inputs.keys())

    def session_handle_for(value, signature_def_input):
        if isinstance(value, (int, float)):
            return None
        # TODO(adamb) This might append to the default graph. These appends
        #     should be cached and idempotent. The added nodes should be
        #     reused if possible.
        v = as_tensor(
                value,
                tf.TensorShape(signature_def_input.tensor_shape).as_list(),
                tf.as_dtype(signature_def_input.dtype))
        if v is None:
            return v
        return tf.get_session_handle(v)

    def _impl(self, **kwargs):
        feed_dict = {}
        session = self.__tfi_session__
        with session.graph.as_default():
            # Cache whether or not we've initialized vars for this method + instance.
            # If not, lazily initialize any vars we'll need.
            if _impl not in self.__tfi_vars_initialized__:
                print("Variables may not yet initialized for", _impl, "out of", var_list)
                uninit_op = tf.report_uninitialized_variables(var_list)
                uninit_op.mark_used()
                uninit_var_names = set(uninit_op.eval(session=session))
                print("Variables not yet initialized:", uninit_var_names)
                uninit_vars = [var for var in var_list if var.name[:-2].encode('utf-8') in uninit_var_names]
                print("Will init:", uninit_vars)
                session.run(tf.variables_initializer(uninit_vars))
                print("Remembering init for", uninit_vars)
                self.__tfi_vars_initialized__.add(_impl)

            handle_keys = []
            handle_in = []
            for key in input_tensor_names_sorted:
                input = signature_def.inputs[key]
                value = kwargs[key]
                # print('finding session_handle_for', key, value, input)
                sh = session_handle_for(value, input)
                # print('found session_handle_for', sh)
                if sh is None:
                    feed_dict[input.name] = value
                else:
                    handle_keys.append(input.name)
                    handle_in.append(sh)

            for key, fh in zip(handle_keys, session.run(handle_in)):
                feed_dict[key] = fh

            # print('feed_dict', feed_dict)
            result = session.run(
                output_tensor_names_sorted,
                feed_dict=feed_dict)

        return dict(zip(output_tensor_keys_sorted, result))

    # Need to properly forge method parameters, complete with annotations.
    argnames = input_tensor_names_sorted
    argdef = ','.join(['_', *argnames])
    argcall = ','.join(['_', *['%s=%s' % (k, k) for k in argnames]])
    gensrc = """lambda %s: _impl(%s)""" % (argdef, argcall)
    impl = eval(gensrc, {'_impl': _impl})
    sigdef_inputs = signature_def.inputs
    impl.__annotations__ = {
        k: sigdef_inputs[k]
        for k, p in inspect.signature(impl).parameters.items()
        if k in sigdef_inputs
    }
    return types.MethodType(impl, instance)

def _infer_signature_defs(class_dict, instance):
    signature_defs = OrderedDict()
    signature_def_docs = OrderedDict()

    fields = instance.__dict__
    # print("fields", fields)
    raw_methods = {}

    for member_name, member in class_dict.items():
        if member_name.startswith('__'):
            continue
        
        if not inspect.isfunction(member):
            # print("Skipping", member_name)
            continue

        raw_methods[member_name] = member

    resolved_methods = {}
    lazy_instance = None
    def _resolve_attr(name):
        if name in fields:
            return fields[name]
        if name not in resolved_methods:
            # HACK(adamb) Bind as method so we can eliminate the initial self parameter.
            raw_method = types.MethodType(raw_methods[name], lazy_instance)
            resolved_methods[name] = _resolve_instance_method_tensors(lazy_instance, raw_method)
        return resolved_methods[name]
    lazy_instance = LazyObject(_resolve_attr)

    for method_name in raw_methods.keys():
        resolved_method = _resolve_attr(method_name)
        signature_def_docs[method_name] = resolved_method.doc
        signature_defs[method_name] = resolved_method.build_signature_def()
    
    return signature_defs, signature_def_docs

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
            @functools.wraps(init)
            def wrapped_init(self, *a, **k):
                if not hasattr(self, '__tfi_hyperparameters__'):
                    hparam_docs = {}
                    if hasattr(init, '__doc__') and init.__doc__:
                        doc = GoogleDocstring(obj=init).result()
                        for hparam_name, hparam_type, hparam_doc in doc['args']:
                            hparam_docs[hparam_name] = hparam_doc

                    ba = inspect.signature(init).bind(self, *a, **k)
                    ba.apply_defaults()
                    self.__tfi_hyperparameters__ = [
                        (hparam_name, type(hparam_val), hparam_val, hparam_docs.get(hparam_name, []))
                        for hparam_name, hparam_val in list(ba.arguments.items())[1:] # ignore self
                    ]

                self.__tfi_graph__ = tf.Graph()
                config = tf.ConfigProto(
                    device_count={'CPU' : 1, 'GPU' : 0},
                    allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options={'allow_growth': True},
                )
                self.__tfi_session__ = tf.Session(graph=self.__tfi_graph__, config=config)
                with self.__tfi_graph__.as_default():
                    with self.__tfi_session__.as_default():
                        init(self, *a, **k)

                        # Once init has executed, we can bind proper methods too!
                        if not hasattr(self, '__tfi_signature_defs__'):
                            self.__tfi_signature_defs__, self.__tfi_signature_def_docs__ = _infer_signature_defs(d, self)

                # print("__tfi_signature_defs__", list(self.__tfi_signature_defs__.keys()))

                    if not hasattr(self, '__tfi_vars_initialized__'):
                        self.__tfi_vars_initialized__ = set()

                    all_var_ops = {
                        v.op: v
                        for v in [*tf.global_variables(), *tf.local_variables()]
                    }
                        # print("all_var_ops", all_var_ops)
                    for method_name, sigdef in self.__tfi_signature_defs__.items():
                        method_vars = set()
                        def visit_method_node(o):
                            if isinstance(o, tf.Operation):
                                if o in all_var_ops:
                                    method_vars.add(all_var_ops[o])
                                # else:
                                #     print("OP NOT A VAR", o.name)
                            elif o.op in all_var_ops:
                                method_vars.add(all_var_ops[o.op])
                            # else:
                            #     print("OTHER NOT A VAR", o.name, o.op, )

                        graph = self.__tfi_graph__
                        _walk([graph.get_tensor_by_name(ti.name) for ti in sigdef.outputs.values()],
                            [graph.get_tensor_by_name(ti.name) for ti in sigdef.inputs.values()],
                            visit_method_node)

                        setattr(self,
                                method_name,
                                _make_method(self, sigdef, method_vars))

            if hasattr(init, '__doc__'):
                wrapped_init.__doc__ = init.__doc__
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
    def __tfi_refresh__(self):
        if not hasattr(self, '__tfi_hyperparameters__'):
            return

        previous_session = self.__tfi_session__
        with previous_session.graph.as_default():
            previous_vars = {
                *tf.global_variables(),
                *tf.local_variables(),
                *tf.model_variables(),
            }
            uninit_op = tf.report_uninitialized_variables(list(previous_vars))
            uninit_op.mark_used()
            uninit_var_names = set(uninit_op.eval(session=previous_session))
            init_vars = [var for var in previous_vars if var.name[:-2].encode('utf-8') not in uninit_var_names]
            previous_saved_prefix = None
            if init_vars:
                previous_saver = tf.train.Saver(var_list=init_vars)
                previous_saved_prefix = previous_saver.save(previous_session, '/tmp/tf-saving')
                print('previous_saved_prefix', previous_saved_prefix)

        # Delete everything that might accidentally pollute our new version, but keep a backup.
        backup = dict(self.__dict__)
        for attr in backup.keys():
            if attr in ['__tfi_hyperparameters__', '__tfi_vars_initialized__']:
                continue
            # print("Removing", attr)
            delattr(self, attr)

        try:
            init_kw = {
                name: val
                for (name, _, val, _) in self.__tfi_hyperparameters__
            }
            self.__init__(**init_kw)
        except:
            # Undo our mess!
            self.__dict__.clear()
            self.__dict__.update(backup)
            raise

        # TODO(adamb) Need to checkpoint all variables in previous_session and
        #     load them into the current one.
        if previous_saved_prefix:
            with self.__tfi_session__.graph.as_default():
                latest_vars = {
                    *tf.global_variables(),
                    *tf.local_variables(),
                    *tf.model_variables(),
                }
                print('latest_vars', latest_vars)
                if latest_vars:
                    latest_saver = tf.train.Saver()
                    latest_saver.restore(self.__tfi_session__, previous_saved_prefix)

def as_class(saved_model_dir, tag_set=tf.saved_model.tag_constants.SERVING):
    from tensorflow.contrib.saved_model.python.saved_model import reader
    from tensorflow.python.saved_model import loader

    def _get_meta_graph_def(saved_model_dir, tag_set):
      saved_model = reader.read_saved_model(saved_model_dir)
      set_of_tags = set(tag_set.split(','))
      for meta_graph_def in saved_model.meta_graphs:
        if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
          return meta_graph_def

      raise RuntimeError('MetaGraphDef associated with tag-set ' + tag_set +
                         ' could not be found in SavedModel')

    # TODO(adamb) Choose a better name than CUSTOMTYPE.
    signature_defs = _get_meta_graph_def(saved_model_dir, tag_set).signature_def
    return type('CUSTOMTYPE', (Base,), {
        '__init__': lambda s:
                loader.load(tf.get_default_session(), tag_set.split(','), saved_model_dir),
        '__tfi_signature_defs__':
                signature_defs,
    })

def export(export_path, model):
    # TODO(adamb) Allow customization of tags.
    tags = [tf.saved_model.tag_constants.SERVING]

    if not isinstance(model, Base):
        raise Exception('%s is not an instance of Base' % model)

    graph = model.__tfi_graph__
    session = model.__tfi_session__

    with graph.as_default():
        # HACK(adamb) For some reason we need to initialize variables before we can save a SavedModel
        vars = {
            *tf.global_variables(),
            *tf.local_variables(),
            *tf.model_variables(),
        }
        uninit_op = tf.report_uninitialized_variables(list(vars))
        uninit_op.mark_used()
        uninit_var_names = set(uninit_op.eval(session=session))
        uninit_vars = [var for var in vars if var.name[:-2].encode('utf-8') in uninit_var_names]
        session.run(tf.variables_initializer(uninit_vars))

        # Don't create this if it already exists. Not sure why we ever need legacy_init_op?
        legacy_init_op = None
        legacy_inits = graph.get_collection(tf.saved_model.constants.LEGACY_INIT_OP_KEY)
        if not len(legacy_inits):
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
              session,
              tags,
              legacy_init_op=legacy_init_op,
              signature_def_map=model.__tfi_signature_defs__)
        builder.save()

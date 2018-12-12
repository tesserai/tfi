from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import inspect
import functools
import os
import re
import threading
import types
import datetime

import tensorflow as tf
import numpy as np

from tfi.base import _GetAttrAccumulator

from tfi.parse.docstring import GoogleDocstring

import tfi.driverconfig.tf
from tfi.driver.tf.tensor_codec import as_tensor

import tfi.tensor.frame
import tfi.driver.tf.checkpoint
import tfi.driver.tf.asset
import tfi.driver.tf.documentation

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
    def __init__(self, doc, input_tensors, output_tensors, output_tensor_shapes, output_tensor_shape_labels, method_name, call_fn, estimator_mode):
        self.doc = doc
        self.estimator_mode = estimator_mode
        self.output_tensor_shapes = output_tensor_shapes
        self.output_tensor_shape_labels = output_tensor_shape_labels
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

    output_tensor_shapes = None
    output_tensor_shape_labels = None
    if is_empty_method(method):
        input_tensors = input_annotations
        output_tensors = _expand_annotation(sig.return_annotation, default={})
        call_fn = _CopyingCall(input_tensors, output_tensors)
    else:
        # If method isn't empty, assume annotations on parameters are kwargs for tf.placeholder
        # Make the resulting graph a bit nicer to read with scope names:
        # <method>/inputs, <method>/_, <method>/outputs.
        drop_output_keys = method.__tfi_drop__ if hasattr(method, '__tfi_drop__') else []
        def _as_output_tensor(name, t):
            if isinstance(t, tf.Operation):
                with tf.control_dependencies([t]):
                    return tf.identity(0, name=name)
            return tf.identity(t, name=name)

        with tf.name_scope(method.__name__):
            with tf.name_scope("inputs"):
                input_tensors = {
                    name: tf.placeholder(name=name, **placeholder_kwargs)
                    for name, placeholder_kwargs in input_annotations.items()
                }
            with tf.name_scope("_"):
                raw_output_tensors = method(**input_tensors)
                output_tensor_shapes = None
                if isinstance(raw_output_tensors, tfi.tensor.frame.TensorFrame):
                    output_tensor_shapes = raw_output_tensors.shapes()
                    output_tensor_shape_labels = raw_output_tensors.shape_labels()
            with tf.name_scope("outputs"):
                output_tensors = {
                    output_name: _as_output_tensor(output_name, t)
                    for output_name, t in raw_output_tensors.items()
                    if output_name not in drop_output_keys
                }
        call_fn = method

    if not output_tensors:
        raise Exception("No output tensors for %s" % method)

    return _RenderedInstanceMethod(
            doc=_enriched_method_doc(input_tensors, output_tensors),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            output_tensor_shapes=output_tensor_shapes,
            output_tensor_shape_labels=output_tensor_shape_labels,
            method_name=method.__tfi_method_name__ if hasattr(method, '__tfi_method_name__') else None,
            estimator_mode=method.__tfi_estimator_mode__ if hasattr(method, '__tfi_estimator_mode__') else None,
            call_fn=call_fn)

def _make_method(instance, signature_def, tensor_shapes, tensor_shape_labels, var_list):
    # Sort to preserve order because we need to go from value to key later.
    output_tensor_keys_sorted = sorted(signature_def.outputs.keys())
    output_tensor_names_sorted = [
        signature_def.outputs[tensor_key].name
        for tensor_key in output_tensor_keys_sorted
    ]

    input_tensor_names_sorted = sorted(signature_def.inputs.keys())

    # TODO(adamb) Stop using session_handles. Use new functionality that allows output tensors to be used in feed_dct for placeholders.

    # If model expects an example, we need to synthesis it.
    def session_handle_for(value, signature_def_input):
        if isinstance(value, (bool, int, float, np.ndarray, bytes, str)):
            return False, None

        # If it's a dict, then use DictToMessage and then SerializeToString or whatever the JSON protobuf thing is...

        # TODO(adamb) This might append to the default graph. These appends
        #     should be cached and idempotent. The added nodes should be
        #     reused if possible.
        tensor_shape = tf.TensorShape(signature_def_input.tensor_shape)
        shape_list = tensor_shape.as_list() if tensor_shape.dims is not None else None
        dtype = tf.as_dtype(signature_def_input.dtype)
        v = as_tensor(value, shape_list, dtype)
        if not isinstance(v, tf.Tensor):
            return False, v
        
        return True, tf.get_session_handle(v)


    def _impl(self, **kwargs):
        feed_dict = {}
        session = self.__tfi_get_session__()
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
            handles = []
            for key in input_tensor_names_sorted:
                input = signature_def.inputs[key]
                value = kwargs[key]
                # print('finding session_handle_for', key, value, input)
                # HACK
                # feed_dict[input.name] = value
                # continue

                is_h, sh = session_handle_for(value, input)
                print('found session_handle_for', sh, type(value))
                if sh is None:
                    feed_dict[input.name] = value
                elif not is_h:
                    feed_dict[input.name] = sh
                else:
                    handle_keys.append(input.name)
                    handle_in.append(sh)

            # print("about to session.run here for", handle_in)
            for key, fh in zip(handle_keys, session.run(handle_in)):
                feed_dict[key] = fh
                handles.append(fh)

            print("about to session.run there", output_tensor_names_sorted, feed_dict, handles)
            result = session.run(
                output_tensor_names_sorted,
                feed_dict=feed_dict)
            # print("result", result)

            for fh in handles:
                tf.delete_session_tensor(fh)

        return tfi.tensor.frame.TensorFrame(*[
            (
                tensor_shapes[tensor_key] if tensor_shapes and tensor_key in tensor_shapes else None,
                tensor_key,
                tensor_result,
            )
            for tensor_key, tensor_result in zip(output_tensor_keys_sorted, result)
        ], **(tensor_shape_labels or {}))

    # Need to properly forge method parameters, complete with annotations.
    argnames = input_tensor_names_sorted
    argdef = ','.join(['', '*', *argnames]) if argnames else ''
    argcall = ','.join(['_', *['%s=%s' % (k, k) for k in argnames]])
    gensrc = """lambda _%s: _impl(%s)""" % (argdef, argcall)
    impl = eval(gensrc, {'_impl': _impl})
    sigdef_inputs = signature_def.inputs
    # NOTE(adamb) Should these annotations be sorted lexigraphically
    impl.__annotations__ = OrderedDict([
        (k, sigdef_inputs[k])
        for k, p in inspect.signature(impl).parameters.items()
        if k in sigdef_inputs
    ])
    sigdef_outputs = signature_def.outputs
    impl.__annotations__['return'] = OrderedDict([
        (k, sigdef_outputs[k])
        for k in sorted(sigdef_outputs.keys())
    ])
    return types.MethodType(impl, instance)

def _infer_signature_defs(class_dict, instance):
    signature_defs = OrderedDict()
    signature_def_docs = OrderedDict()
    estimator_modes = OrderedDict()

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
        estimator_modes[method_name] = resolved_method.estimator_mode

    return signature_defs, signature_def_docs, estimator_modes

class Meta(type):
    @staticmethod
    def __new__(meta, classname, bases, d):
        if '__tfi_del__' in d:
            for name in d['__tfi_del__']:
                del d[name]
            del d['__tfi_del__']

        d['__tfi_refresh_watchers__'] = []

        if '__init__' in d:
            init = d['__init__']
            # Wrap __init__ to graph, session, and methods.
            @functools.wraps(init)
            def wrapped_init(self, *a, **k):
                if '__tfi_graph__' in k:
                    graph = k['__tfi_graph__']
                    del k['__tfi_graph__']
                else:
                    graph = tf.Graph()

                if '__tfi_skip_method_render__' in k:
                    do_method_render = not k['__tfi_skip_method_render__']
                    del k['__tfi_skip_method_render__']
                else:
                    do_method_render = True

                if not hasattr(self, '__tfi_hyperparameters__'):
                    hparam_docs = {}
                    if hasattr(init, '__doc__') and init.__doc__:
                        init_doc = GoogleDocstring(obj=init).result()
                        for hparam_name, hparam_type, hparam_doc in init_doc['args']:
                            hparam_docs[hparam_name] = hparam_doc

                    ba = inspect.signature(init).bind(self, *a, **k)
                    ba.apply_defaults()
                    hparam_items = list(ba.arguments.items())[1:] # ignore self
                    self.__tfi_hyperparameters__ = [
                        (hparam_name, type(hparam_val), hparam_val, hparam_docs.get(hparam_name, []))
                        for hparam_name, hparam_val in hparam_items
                    ]


                if hasattr(self, '__doc__') and self.__doc__:
                    model_doc = GoogleDocstring(obj=self).result()
                    model_doc_sections = model_doc['sections']
                    if not hasattr(self, '__tfi_name__'):
                        self.__tfi_name__ = type(self).__name__

                    if not hasattr(self, '__tfi_overview__'):
                        # NOTE(adamb) Since we don't want to be parsing rst here, we'll just rewrite
                        #     it to include detected citations. Expect that this rst will be parsed
                        #     for real when rendering HTML.
                        text_sections = [v for t, v in model_doc_sections if t == 'text']
                        overview = "\n".join([l for t in text_sections for l in t])
                        self.__tfi_overview__ = overview

                self.__tfi_graph__ = graph
                self.__tfi_session__ = None

                if not hasattr(self, '__tfi_refresh_conditions__'):
                    self.__tfi_refresh_conditions__ = _ConditionGenerator()

                with graph.as_default():
                    with self.__tfi_get_session__().as_default():
                        init(self, *a, **k)

                    # Once init has executed, we can bind proper methods too!
                    if not hasattr(self, '__tfi_signature_defs__'):
                        self.__tfi_signature_defs__, self.__tfi_signature_def_docs__, self.__tfi_estimator_modes__ = _infer_signature_defs(d, self)

                    if not hasattr(self, '__tfi_vars_initialized__'):
                        self.__tfi_vars_initialized__ = set()

                    if do_method_render:
                        all_var_ops = {
                            v.op: v
                            for v in [*tf.global_variables(), *tf.local_variables()]
                        }

                        for method_name, sigdef in self.__tfi_signature_defs__.items():
                            # Should method_var calculation be inside of _make_method?
                            method_vars = set()
                            def visit_method_node(o):
                                if isinstance(o, tf.Operation):
                                    if o in all_var_ops:
                                        method_vars.add(all_var_ops[o])
                                elif o.op in all_var_ops:
                                    method_vars.add(all_var_ops[o.op])

                            _walk([graph.get_tensor_by_name(ti.name) for ti in sigdef.outputs.values()],
                                [graph.get_tensor_by_name(ti.name) for ti in sigdef.inputs.values()],
                                visit_method_node)

                            output_tensor_shapes, output_tensor_shape_labels = self.__tfi_get_signature_def_output_shapes__(method_name)
                            method = _make_method(self, sigdef, output_tensor_shapes, output_tensor_shape_labels, method_vars)
                            setattr(self, method_name, method)

                if not hasattr(self, '__tfi_doc__'):
                    sdd = self.__tfi_signature_defs_docs__
                    self.__tfi_doc__ = tfi.driver.tf.documentation.ModelDocumentation(
                        hyperparameters=self.__tfi_hyperparameters__,
                        name=self.__tfi_name__,
                        overview=self.__tfi_overview__,
                        methods=OrderedDict([
                            (
                                method_name,
                                tfi.driver.tf.documentation.MethodDocumentation(
                                    name=method_name,
                                    overview=sdd[method_name]['sections'],
                                    inputs=sdd[method_name]['args'],
                                    outputs=sdd[method_name]['returns'],
                                    example=tfi.driver.tf.documentation.MethodExample(
                                        inputs={
                                            input_name: eval("\n".join(input_val_lines), {}, {'m': self, 'tfi': tfi})
                                            for input_name, _, input_val_lines in sdd[method_name]['example args']
                                        }
                                    ),
                                ),
                            )
                            for method_name, signature_def in self.__tfi_signature_defs__.items()
                        ]),
                    )

                for fn in self.__tfi_refresh_watchers__:
                    fn(self)

            if hasattr(init, '__doc__'):
                wrapped_init.__doc__ = init.__doc__
            d['__init__'] = wrapped_init

        return super(Meta, meta).__new__(meta, classname, bases, d)

    @staticmethod
    def __prepare__(name, bases):
        def method_name_decorator(name):
            def install_tfi_method_name(fn):
                if hasattr(fn, '__tfi_method_name__'):
                    raise Exception("Already specified @tfi_method_name for %s" % fn)
                fn.__tfi_method_name__ = name
                return fn
            return install_tfi_method_name

        def drop_decorator(keys):
            def install_tfi_drop(fn):
                if hasattr(fn, '__tfi_drop__'):
                    raise Exception("Already specified @tfi_drop for %s" % fn)
                fn.__tfi_drop__ = keys
                return fn
            return install_tfi_drop

        def estimator_mode_decorator(mode):
            def install_tfi_estimator_mode(fn):
                if hasattr(fn, '__tfi_estimator_mode__'):
                    raise Exception("Already specified @tfi_estimator_mode for %s" % fn)
                fn.__tfi_estimator_mode__ = mode
                return fn
            return install_tfi_estimator_mode

        d = OrderedDict({
            'tfi_method_name': method_name_decorator,
            'tfi_estimator_mode': estimator_mode_decorator,
            'tfi_drop': drop_decorator,
            'self': _GetAttrAccumulator(),
        })
        # NOTE(adamb) Remember to delete all of these! Every item in here is
        #     essentially "reserved" and can't be used as a method name in a
        #     SavedModel. Expand it with caution.
        d['__tfi_del__'] = list(d.keys())

        return d

class _ConditionGenerator(object):
    def __init__(self):
        self._c = threading.Condition()

    def notify_all(self):
        with self._c:
            self._c.notify_all()

    def whenever(self):
        while True:
            yield

            with self._c:
                self._c.wait()

class Model(object, metaclass=Meta):
    def __tfi_get_session_logdir__(self, *args, **kwargs):
        self.__tfi_get_session__()
        return self.__tfi_session_logdir_fn__(*args, **kwargs)

    def __tfi_get_signature_def_output_shapes__(self, method_name):
        signature_def = self.__tfi_signature_defs__[method_name]
        return (
            {
                output: tuple([dim.name if dim.name else None if ix > 0 else 'B' for ix, dim in enumerate(tensor_info.tensor_shape.dim)])
                for output, tensor_info in signature_def.outputs.items()
            },
            {},
        )
    
    def __tfi_get_session__(self):
        if not self.__tfi_session__:
            graph = self.__tfi_graph__
            self.__tfi_session_logdir_fn__ = tfi.driverconfig.tf.make_logdir_fn(datetime.datetime.now())
            self.__tfi_session__ = tfi.driverconfig.tf.make_session(graph)
        return self.__tfi_session__

    def __tfi_save_initialized_vars__(self, dir):
        if not self.__tfi_session__:
            return [], None

        session = self.__tfi_session__
        with session.graph.as_default():
            vars = {
                *tf.global_variables(),
                *tf.local_variables(),
                *tf.model_variables(),
            }
            uninit_op = tf.report_uninitialized_variables(list(vars))
            uninit_op.mark_used()
            uninit_var_names = set(uninit_op.eval(session=session))
            init_vars = [var for var in vars if var.name[:-2].encode('utf-8') not in uninit_var_names]

            saved_prefix = None
            init_var_names = []
            if init_vars:
                saver = tf.train.Saver(var_list=list(init_vars))
                saved_prefix = saver.save(session, dir)
                print('saved_prefix', saved_prefix)
                init_var_names = {var.name for var in init_vars}

            return init_var_names, saved_prefix

    def __tfi_restore_vars__(self, saved_prefix, var_filter):
        with self.__tfi_graph__.as_default():
            all_vars = {
                *tf.global_variables(),
                *tf.local_variables(),
                *tf.model_variables(),
            }
            # Only restore vars that still exist *and* were previously initialized.
            vars = [var for var in all_vars if var_filter(var)]
            print("might restore", all_vars, "from", saved_prefix)
            if vars:
                print("restoring", vars, "from", saved_prefix)
                saver = tf.train.Saver(var_list=vars)
                saver.restore(self.__tfi_get_session__(), saved_prefix)

    def __tfi_hyperparameters_dict__(self):
        return {
            name: val
            for (name, _, val, _) in self.__tfi_hyperparameters__
        }

    def __tfi_refreshes__(self):
        for _ in self.__tfi_refresh_conditions__.whenever():
            yield

    def __tfi_refresh__(self):
        if not hasattr(self, '__tfi_hyperparameters__'):
            return

        logdir = os.path.join(self.__tfi_get_session_logdir__(), 'refresh')
        previous_init_var_names, previous_saved_prefix = self.__tfi_save_initialized_vars__(logdir)
        previous_session = self.__tfi_session__

        # Delete everything that might accidentally pollute our new version, but keep a backup.
        backup = dict(self.__dict__)
        for attr in backup.keys():
            if attr in [
                '__tfi_hyperparameters__',
                '__tfi_vars_initialized__',
                '__tfi_refresh_conditions__',
            ]:
                continue
            delattr(self, attr)

        try:
            # Compute init kwargs based on the hyperparameters needed by the latest one
            hparams = self.__tfi_hyperparameters_dict__()
            empty = inspect.Signature.empty
            init_kw = {
                name: hparams.get(name, param.default if param.default != empty else None)
                for name, param in inspect.signature(self.__init__).parameters.items()
            }
            # Compute init kwargs based on existing hyperparameters
            self.__init__(**init_kw)
        except:
            # Undo our mess!
            self.__dict__.clear()
            self.__dict__.update(backup)
            raise

        if previous_saved_prefix and previous_init_var_names:
            self.__tfi_restore_vars__(
                    previous_saved_prefix,
                    lambda var: var.name in previous_init_var_names)

        if previous_session:
            previous_session.close()

        # Notify anyone waiting for refreshes of this model.
        self.__tfi_refresh_conditions__.notify_all()

def as_class(saved_model_path, tag_set=tf.saved_model.tag_constants.SERVING):
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

    def _read_signature_defs(saved_model_dir):
        return _get_meta_graph_def(saved_model_dir, tag_set).signature_def

    tempdirs = []
    if os.path.isdir(saved_model_path):
        saved_model_dir = saved_model_path
    else:
        import tempfile
        import zipfile
        tempdir = tempfile.TemporaryDirectory()
        saved_model_dir = tempdir.name
        tempdirs.append(tempdir)
        with zipfile.ZipFile(saved_model_path) as zipf:
            # print("extracting", zipf.namelist(), "to", saved_model_dir)
            zipf.extractall(saved_model_dir)
            while True:
                extracted_entries = os.listdir(saved_model_dir)
                if len(extracted_entries) == 1:
                    saved_model_dir = os.path.join(saved_model_dir, extracted_entries[0])
                else:
                    break

    signature_defs = _read_signature_defs(saved_model_dir)
    
    facets_overview_statistics_proto = None
    facets_overview_statistics_path = os.path.join(saved_model_dir, 'assets.extra/doc/facets_overview_feature_statistics.pb')
    if os.path.exists(facets_overview_statistics_path):
        with open(facets_overview_statistics_path, 'rb') as f:
            facets_overview_statistics_proto = f.read()


    classname = os.path.basename(saved_model_path)

    doc = tfi.driver.tf.documentation.read(saved_model_dir, signature_defs)
    return type(classname, (Model,), {
        '__init__': lambda s:
                loader.load(s.__tfi_get_session__(), tag_set.split(','), saved_model_dir),
        '__name__': doc.name(),
        '__doc__': doc.docstring(),
        '__tfi_doc__': doc,

        '__tfi_name__': doc.name(),
        '__tfi_overview__': doc.overview(),
        '__tfi_hyperparameters__': doc.hyperparameters(),
        '__tfi_signature_defs_docs__': {
            method_name: {
                'name': method_name,
                'sections': method_doc.overview(),
                'args': method_doc.inputs(),
                'returns': method_doc.outputs(),
                'example args': [
                    (
                        input_name,
                        None,
                        [input_repr],
                    )
                    for input_name, input_repr in method_doc.example().input_reprs().items()
                ],
            }
            for method_name, method_doc in doc.methods().items()
        },

        '__tfi_signature_defs__': signature_defs,
        '__tfi_facets_overview_proto__': facets_overview_statistics_proto,
        '__tfi_tempdirs__': tempdirs,
        '__tfi_estimator_modes__': {
            # For now we assume that *all* methods in a SavedModel should be used for estimator inference
            method_name: 'infer'
            for method_name in signature_defs.keys()
        },
        '__tfi_refresh_conditions__': _ConditionGenerator(),
    })

def load(saved_model_path):
    cls = as_class(saved_model_path)
    return cls()

def dump(export_path, model):
    if export_path.endswith(".zip"):
        export_zip_path = export_path
        export_path = export_path[:-4]
    else:
        export_zip_path = None

    # TODO(adamb) Allow customization of tags.
    tags = [tf.saved_model.tag_constants.SERVING]

    if not isinstance(model, Model):
        raise Exception('%s is not an instance of Model' % model)

    graph = model.__tfi_graph__
    session = model.__tfi_get_session__()

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

        tfi.driver.tf.documentation.write(
            export_path,
            model.__tfi_doc__,
        )

    if export_zip_path:
        import zipfile
        with zipfile.ZipFile(export_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    zipf.write(filepath, arcname=os.path.relpath(filepath, export_path))

export = dump

class _SessionEndHook(tf.train.SessionRunHook):
    def __init__(self, end_hook):
        self._end_hook = end_hook

    def end(self, session):
        self._end_hook(session)

def _estimator_method_name(estimator_modes, mode):
    matching = []
    for method_name, _mode in estimator_modes.items():
        if _mode == mode:
            matching.append(method_name)

    if not matching:
        raise Exception("No estimator configuration for mode %s. Have: %s" % (mode, set(estimator_modes.values())))

    if len(matching) > 1:
        raise Exception("Found multiple matching estimator configurations for mode %s for model %s: %s" % (mode, model, matching))

    return matching[0]

def estimator_method(model, mode):
    return getattr(model, _estimator_method_name(model.__tfi_estimator_modes__, mode))

def as_estimator(model_or_class, model_dir=None):
    def _make_model_fn_from_class(c, estimator_modes, hooks):
        def model_fn(features, labels, mode, params, config):
            """
            Args:
            features: This is the first item returned from the input_fn passed to
                        train, evaluate, and predict. This should be a single Tensor
                        or dict of same.
            labels: This is the second item returned from the input_fn passed to
                    train, evaluate, and predict. This should be a single Tensor or
                    dict of same (for multi-head models). If mode is ModeKeys.PREDICT,
                    labels=None will be passed. If the model_fn's signature does not
                    accept mode, the model_fn must still be able to handle labels=None.
            mode: Optional. Specifies if this training, evaluation or prediction.
                    See ModeKeys.
            params: Optional dict of hyperparameters. Will receive what is passed
                    to Estimator in params parameter. This allows to configure
                    Estimators from hyper parameter tuning.
            config: Optional configuration object. Will receive what is passed to
                    Estimator in config parameter, or the default config. Allows
                    updating things in your model_fn based on configuration such
                    as num_ps_replicas, or model_dir.
            Returns:
            EstimatorSpec
            """
            # NOTE(adamb) We MUST wait to create instance until we're within model_fn, since
            # the Estimator implementation defines its own Graph and Session.
            call_fn = getattr(c, _estimator_method_name(estimator_modes, mode))

            i = c(
                __tfi_graph__=tf.get_default_graph(),
                __tfi_skip_method_render__=True,
                **params)

            call_args = [i]
            call_kwargs = {}
            if isinstance(features, tfi.tensor.frame.TensorFrame):
                features = features.dict()
            if isinstance(features, dict):
                for k, v in features.items():
                    call_kwargs[k] = v
            elif labels is None:
                call_args = [features]
            else:
                call_kwargs['features'] = features

            if isinstance(labels, tfi.tensor.frame.TensorFrame):
                labels = labels.dict()
            if isinstance(labels, dict):
                for k, v in labels.items():
                    call_kwargs[k] = v
            elif labels is not None:
                call_kwargs['labels'] = labels

            r = call_fn(*call_args, **call_kwargs)

            is_train = mode == tf.estimator.ModeKeys.TRAIN
            is_eval = mode == tf.estimator.ModeKeys.EVAL
            is_predict = mode == tf.estimator.ModeKeys.PREDICT

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=r if is_predict else None,
                loss=r['loss'] if is_train or is_eval else None,
                train_op=r['train_op'] if is_train else None,
                eval_metric_ops=r['eval_metric_ops'] if 'eval_metric_ops' in r and (is_train or is_eval) else None,
                training_hooks=hooks.get('train', None),
                evaluation_hooks=hooks.get('eval', None),
                prediction_hooks=hooks.get('infer', None),
            )

        return model_fn

    def _make_estimator_from_instance(instance):
        logdir = os.path.join(instance.__tfi_get_session_logdir__(), 'estimator')

        saved_var_names, saved_prefix = instance.__tfi_save_initialized_vars__(logdir)
        warm_start_from = None
        if saved_prefix:
            warm_start_from = tf.estimator.WarmStartSettings(
                saved_prefix,
                vars_to_warm_start=re.compile("|".join([re.escape(var_name) for var_name in saved_var_names])))

        config = tf.estimator.RunConfig(
            model_dir=model_dir or logdir,
            tf_random_seed=None,
            session_config=None,
            keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=10000,
            log_step_count_steps=100,
        )

        estimator = None
        def train_hook(session):
            trained_var_names = set(estimator.get_variable_names())
            print("will try to restore", trained_var_names)
            instance.__tfi_restore_vars__(
                estimator.latest_checkpoint(),
                lambda var: var.op.name in trained_var_names)

        # We need to save a reference to the Estimator so we can export variables post training.
        estimator = tf.estimator.Estimator(
            model_fn=_make_model_fn_from_class(
                    instance.__class__,
                    instance.__tfi_estimator_modes__,
                    hooks={
                        'train': [_SessionEndHook(train_hook)],
                    }),
            config=config,
            params=instance.__tfi_hyperparameters_dict__(),
            warm_start_from=warm_start_from,
        )
        return estimator

    if not isinstance(model_or_class, tfi.driver.tf.Model):
        raise Exception('%s is not an instance of Model' % instance)

    return _make_estimator_from_instance(model_or_class)


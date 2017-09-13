from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import warnings

import types
import collections

import numpy as np

import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader

from tfi.as_tensor import as_tensor

def _get_meta_graph_def(saved_model_dir, tag_set):
  """Gets MetaGraphDef from SavedModel.
  Returns the MetaGraphDef for the given tag-set and SavedModel directory.
  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,
        separated by ','. For tag-set contains multiple tags, all tags must be
        passed in.
  Raises:
    RuntimeError: An error when the given tag-set does not exist in the
        SavedModel.
  Returns:
    A MetaGraphDef corresponding to the tag-set.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def

  raise RuntimeError('MetaGraphDef associated with tag-set ' + tag_set +
                     ' could not be found in SavedModel')

def as_class(
        saved_model_dir,
        tag_set=tf.saved_model.tag_constants.SERVING,
        tf_debug=False):
    classname = 'CUSTOMTYPE'
    def _make_method(signature_def_name, signature_def):
        # Sort to preserve order because we need to go from value to key later.
        output_tensor_keys_sorted = sorted(signature_def.outputs.keys())
        output_tensor_names_sorted = [
            signature_def.outputs[tensor_key].name
            for tensor_key in output_tensor_keys_sorted
        ]

        input_tensor_names_sorted = sorted(signature_def.inputs.keys())

        result_class_name = '%s__%s__Result' % (classname, signature_def_name)
        result_class = collections.namedtuple(result_class_name, output_tensor_keys_sorted)

        def session_handle_for(value, signature_def_input):
            return as_tensor(
                value,
                tf.TensorShape(signature_def_input.tensor_shape).as_list(),
                tf.as_dtype(signature_def_input.dtype))

        def _run(self, **feed_dict):
            with self._sess.graph.as_default():
                feed_handles = {
                    signature_def.inputs[key].name: fh
                    for key, fh in zip(
                        input_tensor_names_sorted,
                        self._sess.run(
                            [
                                session_handle_for(feed_dict[key], signature_def.inputs[key])
                                for key in input_tensor_names_sorted
                            ]))
                }

                result = self._sess.run(
                    output_tensor_names_sorted,
                    feed_dict=feed_handles)

            return result_class._make(result)
        return _run

    def __init__(self):
        sess = session.Session(graph=ops_lib.Graph())
        loader.load(sess, tag_set.split(','), saved_model_dir)
        if tf_debug:
          sess = local_cli_wrapper.LocalCLIDebugWrapperSession(sess)
        self._sess = sess

    meta_graph_def = _get_meta_graph_def(saved_model_dir, tag_set)
    classdict = {
        name: _make_method(name, signature_def)
        for name, signature_def in meta_graph_def.signature_def.items()
    }
    classdict['__init__'] = __init__

    return type(classname, (object,), classdict)

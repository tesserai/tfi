import tfi.json
import tensorflow as tf
import os.path
import numpy as np

import tfi.data

from collections import OrderedDict

from google.protobuf.json_format import ParseDict


class ModelDocumentation(object):
    def __init__(self, *, hyperparameters, name, overview, methods):
        self._hyperparameters = hyperparameters
        self._name = name
        self._overview = overview
        self._methods = methods

    def hyperparameters(self): return self._hyperparameters

    def name(self): return self._name

    def overview(self): return self._overview

    def methods(self): return self._methods

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr

class MethodDocumentation(object):
    def __init__(self, *, name, overview, inputs, outputs, example):
        self._name = name # str
        self._overview = overview # [str, ...]
        self._inputs = inputs # [(name, tensor_info_str, doc: [str])]
        self._outputs = outputs # [(name, tensor_info_str, doc: [str])]
        self._example = example

    def name(self): return self._name

    def overview(self): return self._overview

    def inputs(self): return self._inputs

    def outputs(self): return self._outputs

    def example(self): return self._example

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr

class MethodExample(object):
    def __init__(self, *, inputs):
        self._inputs = inputs

    def inputs(self): return self._inputs

    def input_reprs(self):
         return {
             k: self._repr(v)
             for k, v in self._inputs.items()
         }

    def _repr(self, tensor_value):
        if isinstance(tensor_value, tf.SparseTensorValue):
            # THIS IS VERY WRONG. ASSUMES A RAGGED SPARSE TENSOR.
            return self._repr(tensor_value.values)
        if isinstance(tensor_value, np.ndarray):
            if tensor_value.dtype.kind == 'O':
                tensor_value = np.vectorize(lambda x: x.decode('utf-8'))(tensor_value)
            return repr([tensor_value.tolist()]) 

        return repr(tensor_value)


class MethodDocumentationLayout(object):
    def __init__(self, base_path, assets_extra_path):
        self.assets_extra_path = assets_extra_path
        self.metadata_path = os.path.join(base_path, 'metadata.json')
        self._base_path = base_path

    def file(self, subpath):
        return os.path.join(self._base_path, subpath)

class ModelDocumentationLayout(object):
    def __init__(self, model_dir):
        self.basename = os.path.basename(model_dir)
        self.assets_extra_path = os.path.join(model_dir, 'assets.extra')
        self.doc_path = os.path.join(self.assets_extra_path, 'doc')
        self.metadata_path = os.path.join(self.doc_path, 'metadata.json')
        self.methods_path = os.path.join(self.doc_path, 'methods')

    def method(self, method_name):
        return MethodDocumentationLayout(
            os.path.join(self.methods_path, method_name),
            self.assets_extra_path,
        )

def _read_json_else(path, default):
    if not os.path.exists(path):
      return default

    with open(path) as f:
      return tfi.json.load(f)

def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        tfi.json.dump(obj, f)

class MethodExampleCodec(object):
    def __init__(self, method_layout):
        self._layout = method_layout

    def write(self, method_example):
        print("method_example.inputs", method_example.inputs())
        _write_json(
            self._layout.file('inputs.json'),
            {
                name: value
                for name, value in method_example.inputs().items()
            }
        )

    def read(self, signature_def):
        return MethodExample(
            inputs=self._detect(
                lambda: self._read_json_tf_example_from(
                    signature_def.inputs,
                    'inputs.pb.json',
                ),
                lambda: self._read_json_example_from(
                    signature_def.inputs,
                    'inputs.json',
                ),
            ),
            # 'example result': _detect(
            #     lambda: _read_json_tf_example_from(
            #         signature_def.outputs,
            #         'outputs.pb.json',
            #     ),
            #     lambda: _read_json_example_from(
            #         signature_def.outputs,
            #         'outputs.json',
            #     ),
            # ),
        )

    def _detect(self, *fns):
        for fn in fns:
            result = fn()
            if result is not None:
                return result

    def _feature_for_tensor_info(self, tensor_info):
        tensor_shape = tensor_info.tensor_shape.dim[1:]
        dtype = tf.DType(tensor_info.dtype)
        if tensor_shape[-1].size != -1:
            return tf.FixedLenFeature(dtype=dtype, shape=[dim.size for dim in tensor_shape])
        return tf.VarLenFeature(dtype=dtype)

    def _read_json_tf_example_from(self, tensor_infos, subpath):
        path = self._layout.file(subpath)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            example_dict = tfi.json.load(f)

        with tf.Session(graph=tf.Graph()) as session:
            example_features = {
                name: self._feature_for_tensor_info(tensor_info)
                for name, tensor_info in tensor_infos.items()
            }
            return session.run(
                tf.parse_single_example(
                    ParseDict(example_dict, tf.train.Example()).SerializeToString(),
                    features=example_features))

    def _read_json_example_from(self, tensor_infos, subpath):
        path = self._layout.file(subpath)
        if not os.path.exists(path):
            return None

        with open(path) as f:
            return tfi.data.json(
                f.read(),
                assets_extra_root=self._layout.assets_extra_path)

class MethodDocumentationCodec(object):
    def __init__(self, method_name, method_layout):
        self._name = method_name
        self._layout = method_layout

    def write(self, method_doc):
        metadata = {
            'documentation': {
                'inputs': {
                    name: doc
                    for name, tensor_info, doc in method_doc.inputs()
                },
                'outputs': {
                    name: doc
                    for name, tensor_info, doc in method_doc.outputs()
                },
            },
        }
        MethodExampleCodec(self._layout).write(method_doc.example()),
        _write_json(self._layout.metadata_path, metadata)
        
    def read(self, signature_def):
        metadata = _read_json_else(self._layout.metadata_path, {})
        doc = metadata.get('documentation', {})
        doc_inputs = doc.get('inputs', {})
        doc_outputs = doc.get('outputs', {})

        return MethodDocumentation(
            name=self._name,
            overview=metadata.get('overview', []),
            inputs=[
                (name, self._tensor_info_str(ti), doc_inputs.get(name, ''))
                for name, ti in signature_def.inputs.items()
            ],
            outputs=[
                (name, self._tensor_info_str(ti), doc_outputs.get(name, ''))
                for name, ti in signature_def.outputs.items()                    
            ],
            example=MethodExampleCodec(self._layout).read(signature_def),
        )

    def _tensor_info_str(self, tensor_info):
        if tensor_info.tensor_shape.unknown_rank:
            return '%s ?' % tf.as_dtype(tensor_info.dtype).name

        return '%s <%s>' % (
            tf.as_dtype(tensor_info.dtype).name,
            ', '.join([
                '?' if dim.size == -1 else str(dim.size)
                for dim in tensor_info.tensor_shape.dim
            ]),
        )

class ModelDocumentationCodec(object):
    def __init__(self, path):
        self._layout = ModelDocumentationLayout(path)

    def _method_codecs(self, method_names):
        return [
            (
                method_name,
                MethodDocumentationCodec(
                    method_name,
                    self._layout.method(method_name),
                )
            )
            for method_name in method_names
        ]

    def write(self, model_doc):
        metadata = {
            'name': model_doc.name(),
            'overview': model_doc.overview(),
            'hyperparameters': [
                (name, str(val_type), val, docs)
                for name, val_type, val, docs in model_doc.hyperparameters()
            ],
        }

        methods = model_doc.methods()
        for method_name, method_codec in self._method_codecs(methods.keys()):
            method_codec.write(methods[method_name])

        _write_json(self._layout.metadata_path, metadata)

    def read(self, signature_defs):
        metadata = _read_json_else(self._layout.metadata_path, {})
        return ModelDocumentation(
            # TODO(adamb) Should be transformed to the below structure, with val_type_str -> val_type
            # (name, val_type, val, docs)
            hyperparameters=metadata.get('hyperparameters', []),
            name=metadata.get('name', self._layout.basename),
            overview=metadata.get('overview', None),
            methods=OrderedDict([
                (method_name, method_codec.read(signature_defs[method_name]))
                for method_name, method_codec in self._method_codecs(signature_defs.keys())
            ]),
        )

def read(path, signature_defs):
    return ModelDocumentationCodec(path).read(signature_defs)

def write(path, model_doc):
    return ModelDocumentationCodec(path).write(model_doc)
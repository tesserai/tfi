import tfi.json
import tensorflow as tf
import os.path

import tfi.data
import tfi.doc

from google.protobuf.json_format import ParseDict
from tfi.parse.docstring import GoogleDocstring

def _detect_method_documentation(*, bibliographer, model, method_name, method, signature_def):
    # NOTE(adamb) Since we don't want to be parsing rst here, we'll just rewrite
    #     it to include detected citations. Expect that this rst will be parsed
    #     for real when rendering HTML.
    docstr = GoogleDocstring(obj=method).result()
    docstr_sections = docstr['sections']
    text_sections = [v for k, v in docstr_sections if k == 'text']
    overview = "\n".join([l for t in text_sections for l in t])

    docstr['args'] = _enrich_docs_with_tensor_info(docstr['args'], signature_def.inputs)
    docstr['returns'] = _enrich_docs_with_tensor_info(docstr['returns'], signature_def.outputs)

    return tfi.doc.MethodDocumentation(
        name=method_name,
        overview=bibliographer.rewrite(overview),
        inputs=docstr['args'],
        outputs=docstr['returns'],
        examples=[
            tfi.doc.MethodDataDocumentation.generate(
                method=getattr(model, method_name),
                inputs={
                    input_name: eval("\n".join(input_val_lines), {}, {'m': model, 'tfi': tfi})
                    for input_name, _, input_val_lines in docstr['example args']
                },
            ),
        ],
    )

def detect_model_documentation(model):
        source = tfi.doc.ModelSource.detect(model)

        bibliographer = tfi.doc.Bibliographer()

        def maybeattr(o, attr, default=None):
            return getattr(o, attr) if o and hasattr(o, attr) else default

        # NOTE(adamb) Since we don't want to be parsing rst here, we'll just rewrite
        #     it to include detected citations. Expect that this rst will be parsed
        #     for real when rendering HTML.
        model_docstr = GoogleDocstring(obj=model).result()
        model_docstr_sections = model_docstr['sections']
        text_sections = [v for k, v in model_docstr_sections if k == 'text']
        overview = "\n".join([l for t in text_sections for l in t])

        return tfi.doc.ModelDocumentation(
            name=maybeattr(model, '__name__', type(model).__name__),
            hyperparameters=maybeattr(model, '__tfi_hyperparameters__', []),
            overview=bibliographer.rewrite(overview),
            implementation_notes=[],
            authors=[
                *[
                    {
                        "name": author['name'],
                        "url": author['url'],
                        "role_noun": "Commits",
                        "role_url": author['commits_url'],
                    }
                    for author in maybeattr(source, 'authors', [])
                ],
            ],
            source=source,
            facets_overview_proto=maybeattr(model, '__tfi_facets_overview_proto__'),
            methods=[
                _detect_method_documentation(
                    model=model,
                    bibliographer=bibliographer,
                    method_name=method_name,
                    method=getattr(model, method_name),
                    signature_def=signature_def,
                )
                for method_name, signature_def in maybeattr(model, '__tfi_signature_defs__').items()
            ],
            references=bibliographer.references(),
        )

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

class MethodDataDocumentationCodec(object):
    def __init__(self, method_layout):
        self._layout = method_layout

    def write(self, method_example):
        _write_json(
            self._layout.file('inputs.json'),
            {
                name: value
                for name, value in method_example.inputs().items()
            }
        )

        if method_example.outputs() is not None:
            _write_json(
                self._layout.file('outputs.json'),
                {
                    name: value
                    for name, value in method_example.outputs().items()
                }
            )

    def read(self, signature_def):
        return tfi.doc.MethodDataDocumentation(
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
            outputs=self._detect(
                lambda: self._read_json_tf_example_from(
                    signature_def.outputs,
                    'outputs.pb.json',
                ),
                lambda: self._read_json_example_from(
                    signature_def.outputs,
                    'outputs.json',
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
        MethodDataDocumentationCodec(self._layout).write(method_doc.examples()[0]),
        _write_json(self._layout.metadata_path, metadata)
        
    def read(self, signature_def):
        metadata = _read_json_else(self._layout.metadata_path, {})
        doc = metadata.get('documentation', {})
        doc_inputs = doc.get('inputs', {})
        doc_outputs = doc.get('outputs', {})

        return tfi.doc.MethodDocumentation(
            name=self._name,
            overview=metadata.get('overview', None),
            inputs=[
                (name, self._tensor_info_str(ti), doc_inputs.get(name, ''))
                for name, ti in signature_def.inputs.items()
            ],
            outputs=[
                (name, self._tensor_info_str(ti), doc_outputs.get(name, ''))
                for name, ti in signature_def.outputs.items()                    
            ],
            examples=[
              MethodDataDocumentationCodec(self._layout).read(signature_def),
            ],
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
            'authors': model_doc.authors(),
            'references': model_doc.references(),
            'implementation_notes': model_doc.implementation_notes(),
            'source': model_doc.source(),
            'facets_overview_proto': None, # model_doc.facets_overview_proto(),
        }

        methods = model_doc.methods()
        for method_name, method_codec in self._method_codecs(methods.keys()):
            method_codec.write(methods[method_name])

        _write_json(self._layout.metadata_path, metadata)

    def read(self, signature_defs):
        metadata = _read_json_else(self._layout.metadata_path, {})
        return tfi.doc.ModelDocumentation(
            # TODO(adamb) Should be transformed to the below structure, with val_type_str -> val_type
            # (name, val_type, val, docs)
            hyperparameters=metadata.get('hyperparameters', []),
            name=metadata.get('name', self._layout.basename),
            overview=metadata.get('overview', None),
            methods=[
                method_codec.read(signature_defs[method_name])
                for method_name, method_codec in self._method_codecs(signature_defs.keys())
            ],
            authors=metadata.get('authors', []),
            references=metadata.get('references', {}),
            implementation_notes=metadata.get('implementation_notes', []),
            source=metadata.get('source', []),
            facets_overview_proto=None,
        )

def read(path, signature_defs):
    return ModelDocumentationCodec(path).read(signature_defs)

def write(path, model_doc):
    return ModelDocumentationCodec(path).write(model_doc)
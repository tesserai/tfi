import tfi.json
import tensorflow as tf
import os.path
import numpy as np

import tfi.data

from collections import OrderedDict

from google.protobuf.json_format import ParseDict

from tfi.parse.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.parse.doi import discover_dois as _discover_dois



class ModelSource(object):
    @classmethod
    def detect(cls, model):
        git_authorship_file = None
        if hasattr(model, '__file__'):
            git_authorship_file = model.__file__
        elif hasattr(model, '__tfi_file__'):
            git_authorship_file = model.__tfi_file__
        elif hasattr(model, '__tfi_module__'):
            git_authorship_file = model.__tfi_module__.__file__

        if not git_authorship_file:
            return None
        github_user_repo = _GitUserRepo("github-users.json")
        git = _git_authorship(github_user_repo, git_authorship_file)
        return cls(
            url=git['url'],
            label=git['label'],
            commit=git['commit'][:7],
            authors=git['authors'],
        )

    def __init__(self, *, url, label, commit, authors):
        self._url = url
        self._label = label
        self._commit = commit
        self._authors = authors

    def url(self): self._url

    def label(self): self._label

    def commit(self): self._commit

    def authors(self): self._authors


from tfi.parse.biblib import bib as _bib
from collections import OrderedDict as _OrderedDict

from tfi.resolve.git import git_authorship as _git_authorship
from tfi.resolve.git import GitUserRepo as _GitUserRepo

class CitationResolver(object):
    def __init__(self, bibtex_resolve):
        self._bibtex_resolve = bibtex_resolve
        self._bibtex_parser = _bib.Parser()
        self._bibtex_parse = self._bibtex_parser.parse
        self._citation_ids = {}

    def references(self):
        return _OrderedDict([
            (k, _OrderedDict(v))
            for k, v in reversed(self._bibtex_parser.get_entries().items())
        ])

    def resolve_citation_id(self, id_type, id):
        if id not in self._citation_ids:
            bibtex = self._bibtex_resolve(**{id_type: [id]})[0]
            self._bibtex_parse(bibtex, log_fp=sys.stderr)
            if bibtex.startswith("@article{"):
                self._citation_ids[id] = bibtex.split(",", 1)[0][len("@article{"):]
            else:
                return id

        return self._citation_ids[id]

class Bibliographer(object):
    def __init__(self):
        from tfi.resolve.arxiv2bib import arxiv2bib as _arxiv2bib
        from tfi.resolve.doi2bib import doi2bib as _doi2bib
        from tfi.resolve.bibtex import BibtexRepo as _BibtexRepo

        bibtex_repo = _BibtexRepo("bibtex.json", {"arxiv_id": _arxiv2bib, "doi": _doi2bib})
        self._citation_resolver = CitationResolver(bibtex_resolve=bibtex_repo.resolve)
        self._resolve_citation_id = self._citation_resolver.resolve_citation_id

    def references(self):
        return self._citation_resolver.references()

    def rewrite(self, paragraph):
        # Should return index ranges with ids, along with ids
        # themselves. Then we can replace them directly in paragraph with
        # references to their ids and accumulate an id citation.
        matches = [
            *[
                (span, 'arxiv_id', id)
                for (span, id) in _discover_arxiv_ids(paragraph)
            ],
            *[
                (span, 'doi', id)
                for (span, id) in _discover_dois(paragraph)
            ],
        ]

        # Replace span in reverse with archive_id
        matches.sort(reverse=True)

        new_paragraph_parts = []
        prev_start = len(paragraph)

        # Rewrite paragraph in reverse, so span indexes are correct.
        # Replace URLs to papers with proper references to papers.
        # Accumulate paragraph pieces and then join them at the very
        # end to avoid wasting string allocations.
        for (start, end), id_type, id in matches:
            try:
                citation_id = None
                citation_id = self._resolve_citation_id(id_type, id)
            except Exception as ex:
                print(ex)
                import traceback
                traceback.print_exc()
            suffix = paragraph[end:prev_start]
            new_paragraph_parts.append(suffix)
            if citation_id:
                new_paragraph_parts.append("]_")
                new_paragraph_parts.append(citation_id)
                new_paragraph_parts.append(" [")
            prev_start = start
        if prev_start != 0:
            new_paragraph_parts.append(paragraph[0:prev_start])

        new_paragraph_parts.reverse()
        return "".join(new_paragraph_parts)

from tfi.parse.docstring import GoogleDocstring

class MethodExample(object):
    @classmethod
    def generate(cls, *, method, inputs):
        return cls(
            inputs=inputs,
            outputs=method(**inputs)
        )

    def __init__(self, *, inputs, outputs=None):
        self._inputs = inputs
        self._outputs = outputs

    def inputs(self): return self._inputs

    def input_reprs(self):
         return {
             k: self._repr(v)
             for k, v in self._inputs.items()
         }

    def outputs(self): return self._outputs

    def output_reprs(self):
         return {
             k: self._repr(v)
             for k, v in self._outputs.items()
         } if self._outputs is not None else None

    def _repr(self, tensor_value):
        if isinstance(tensor_value, tf.SparseTensorValue):
            # THIS IS VERY WRONG. ASSUMES A RAGGED SPARSE TENSOR.
            return self._repr(tensor_value.values)
        if isinstance(tensor_value, np.ndarray):
            if tensor_value.dtype.kind == 'O':
                tensor_value = np.vectorize(lambda x: x.decode('utf-8'))(tensor_value)
            return repr([tensor_value.tolist()]) 

        return repr(tensor_value)

    def with_updated_outputs(self, method):
        return MethodExample.generate(
            method=method,
            inputs=self._inputs,
        )

class MethodDocumentation(object):
    @classmethod
    def detect(cls, *, bibliographer, model, method_name, method, signature_def):
        # NOTE(adamb) Since we don't want to be parsing rst here, we'll just rewrite
        #     it to include detected citations. Expect that this rst will be parsed
        #     for real when rendering HTML.
        docstr = GoogleDocstring(obj=method).result()
        docstr_sections = docstr['sections']
        text_sections = [v for k, v in docstr_sections if k == 'text']
        overview = "\n".join([l for t in text_sections for l in t])

        docstr['args'] = _enrich_docs_with_tensor_info(docstr['args'], signature_def.inputs)
        docstr['returns'] = _enrich_docs_with_tensor_info(docstr['returns'], signature_def.outputs)

        return cls(
            name=method_name,
            overview=bibliographer.rewrite(overview),
            inputs=docstr['args'],
            outputs=docstr['returns'],
            example=tfi.driver.tf.documentation.MethodExample.generate(
                method=getattr(model, method_name),
                inputs={
                    input_name: eval("\n".join(input_val_lines), {}, {'m': model, 'tfi': tfi})
                    for input_name, _, input_val_lines in docstr['example args']
                },
            ),
        )

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

    def with_updated_example_outputs(self, method):
        return MethodDocumentation(
            name=self._name,
            overview=self._overview,
            inputs=self._inputs,
            outputs=self._outputs,
            example=self._example.with_updated_outputs(method),
        )

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr


class ModelDocumentation(object):
    @classmethod
    def detect(cls, model):
        source = ModelSource.detect(model)

        bibliographer = Bibliographer()

        def maybeattr(o, attr, default=None):
            return getattr(o, attr) if o and hasattr(o, attr) else default

        # NOTE(adamb) Since we don't want to be parsing rst here, we'll just rewrite
        #     it to include detected citations. Expect that this rst will be parsed
        #     for real when rendering HTML.
        model_docstr = GoogleDocstring(obj=model).result()
        model_docstr_sections = model_docstr['sections']
        text_sections = [v for k, v in model_docstr_sections if k == 'text']
        overview = "\n".join([l for t in text_sections for l in t])

        return tfi.driver.tf.documentation.ModelDocumentation(
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
                tfi.driver.tf.documentation.MethodDocumentation.detect(
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

    def __init__(self, *,
        hyperparameters,
        name,
        overview,
        methods,
        authors,
        references,
        implementation_notes,
        source,
        facets_overview_proto,
    ):
        self._hyperparameters = hyperparameters
        self._name = name
        self._overview = overview
        self._methods = methods
        self._authors = authors
        self._references = references
        self._implementation_notes = implementation_notes
        self._source = source
        self._facets_overview_proto = facets_overview_proto

    def hyperparameters(self): return self._hyperparameters

    def name(self): return self._name

    def overview(self): return self._overview

    def methods(self): return self._methods

    def authors(self): return self._authors

    def references(self): return self._references

    def implementation_notes(self): return self._implementation_notes

    def source(self): return self._source

    def facets_overview_proto(self): return self._facets_overview_proto

    def with_updated_example_outputs(self, model):
        return ModelDocumentation(
            hyperparameters=self._hyperparameters,
            name=self._name,
            overview=self._overview,
            authors=self._authors,
            references=self._references,
            implementation_notes=self._implementation_notes,
            source=self._source,
            facets_overview_proto=self._facets_overview_proto,
            methods=[
                method.with_updated_example_outputs(getattr(model, method.name()))
                for method in self._methods
            ],
        )

    def docstring(self):
        name = self._name
        overview = self._overview
        docstr = None
        if overview:
            docstr = "\n".join([name, "-" * len(name), "", overview])

        return docstr

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

class MethodExampleCodec(object):
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
        MethodExampleCodec(self._layout).write(method_doc.example()),
        _write_json(self._layout.metadata_path, metadata)
        
    def read(self, signature_def):
        metadata = _read_json_else(self._layout.metadata_path, {})
        doc = metadata.get('documentation', {})
        doc_inputs = doc.get('inputs', {})
        doc_outputs = doc.get('outputs', {})

        return MethodDocumentation(
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
        return ModelDocumentation(
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
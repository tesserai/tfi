import spacy
import inspect
import functools
import tempfile
import zipfile
import os
import time
import json

from collections import OrderedDict
from tfi.parse.docstring import GoogleDocstring

def _resolve_instance_method_tensors(instance, fn, docstring=None):
  def _expand_annotation(instance, annotation, default=None):
      if annotation == inspect.Signature.empty:
          return default
      if isinstance(annotation, dict):
        return {
          k: _expand_annotation(instance, v)
          for k, v in annotation.items()
        }
      if isinstance(annotation, type):
        return {'dtype': annotation}
      return annotation

  def _tensor_info_str(tensor):
    if not tensor:
      tensor = {}

    shape_list = tensor.get('shape', [])
    ndims = len(shape_list)
    dtype = tensor.get('dtype', None)
    if dtype is None:
      dtype_name = 'any'
    elif isinstance(dtype, type):
      dtype_name = dtype.__name__
    else:
      dtype_name = str(dtype)

    if ndims is None:
      return "%s ?" % dtype_name

    if len(shape_list) == 0:
      shape = "scalar"
    else:
      shape = "<%s>" % (
          ", ".join(["?" if n is None else str(n) for n in shape_list]),
      )
    return "%s %s" % (dtype_name, shape)

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
    (name, _expand_annotation(instance, param.annotation))
    for name, param in sig.parameters.items()
  ])
  output_annotations = OrderedDict([
    (name, _expand_annotation(instance, value))
    for name, value in _expand_annotation(instance, sig.return_annotation, {}).items()
  ])

  if fn.__doc__ or docstring:
    doc = GoogleDocstring(obj=fn, docstring=docstring).result()
  else:
    doc = {'sections': [], 'args': {}, 'returns': {}}
  doc['args'] = _enrich_docs(doc['args'], input_annotations)
  doc['returns'] = _enrich_docs(doc['returns'], output_annotations)

  return doc, input_annotations, output_annotations

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
                    docstrings = {}
                    if hasattr(self, '__tfi_docstrings__'):
                      docstrings = self.__tfi_docstrings__

                    for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
                        if method_name.startswith('_'):
                            continue

                        docstring = docstrings.get(method_name, None)
                        doc, input_annotations, output_annotations = _resolve_instance_method_tensors(self, method, docstring=docstring)

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


class Model(object, metaclass=Meta):
  def __init__(self, spacy_model, __tfi_tempdirs__=None, __tfi_docstrings__=None):
    self._spacy_model = spacy_model
    meta = spacy_model.meta
    self.__name__ = meta['name']
    self.__doc__ = meta['description']
    self.__tfi_tempdirs__ = __tfi_tempdirs__
    self.__tfi_docstrings__ = __tfi_docstrings__

  def __tfi_init__(self):
    pass

  def entities(self, text: str):
    # TODO(adamb) Delete this comment once SecureDocs switches to newer model with documentation embedded!
    """
    Example args:
      text: 'This EXECUTIVE EMPLOYMENT AND NON-COMPETITION AGREEMENT (this "Agreement"), dated as of June 26, 2006, is entered into by and between ENERGYSOLUTIONS, LLC, a Utah limited liability company (the "Company"), and Val John Christensen (the "Executive").'
    """
    doc = self._spacy_model(text)
    return [
      {
        'text': ent.text,
        'label_': ent.label_,
        'start_char': ent.start_char,
        'end_char': ent.end_char,
      }
      for ent in doc.ents
    ]

def _maybe_squashfs_image(file):
  try:
    import PySquashfsImage
    return PySquashfsImage.SquashFsImage(file)
  except ImportError:
      return None
  except IOError:
    return None

def _import_dir(mktempdir, import_path):
  if os.path.isdir(import_path):
    return import_path

  import zipfile
  if zipfile.is_zipfile(import_path):
    import_dir = mktempdir()
    import subprocess
    print("Trying fuse-zip")
    result = subprocess.run(["fuse-zip", import_path, import_dir])
    if result.returncode:
      print(result.stderr or result.stdout)
      print("Falling back to extracting zip after fuse-zip failed.")
      with zipfile.ZipFile(import_path) as zipf:
        zipf.extractall(import_dir)

    while True:
      extracted_entries = os.listdir(import_dir)
      if len(extracted_entries) == 1:
        return os.path.join(import_dir, extracted_entries[0])
  
  squashfs_image = _maybe_squashfs_image(import_path)
  if squashfs_image:
    squashfs_image.close()
    import subprocess
    import_dir = mktempdir()

    # result = subprocess.run(["mount", import_path, import_dir, "-t", "squashfs", "-o", "loop,ro,noexec,noload"])
    result = subprocess.run(["squashfuse", import_path, import_dir])
    if result.returncode:
      raise Exception(result.stderr or result.stdout)

    return import_dir

def load(import_path):
  tempdirs = []
  def mktempdir():
    import tempfile
    tempdir = tempfile.TemporaryDirectory()
    tempdirs.append(tempdir)
    return tempdir.name

  now = time.time()
  import_dir = _import_dir(mktempdir, import_path)
  print("_import_dir took %ss" % (time.time() - now))

  docstrings = {}
  examples_path = os.path.join(import_dir, "tfi-examples.json")
  if os.path.exists(examples_path):
    with open(examples_path) as examples_f:
      docstrings = {
        method_name: "\n".join(
          [
            "",
            "  Example args:",
            *[
              "    %s: %s" % (k, repr(v))
              for k, v in example.items()
            ],
            ""
          ])
        for method_name, example in json.load(examples_f).items()
      }

  spacy_model = spacy.load(import_dir)
  return Model(spacy_model, __tfi_tempdirs__=tempdirs, __tfi_docstrings__=docstrings)

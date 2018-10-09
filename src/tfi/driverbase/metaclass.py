import inspect
import functools

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
      if not isinstance(annotation, dict) and not hasattr(annotation, '__getitem__') and not hasattr(annotation, 'get'):
        raise Exception("Annotation is not a dict and doesn't support both get and __getitem__: %s" % annotation)
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
                      docstrings = self.__tfi_docstrings__ or docstrings

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


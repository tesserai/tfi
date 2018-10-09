import spacy
import time

import tempfile
import os
import json

from tfi.driverbase import Meta as _Meta, import_dir as _import_dir

class Model(object, metaclass=_Meta):
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

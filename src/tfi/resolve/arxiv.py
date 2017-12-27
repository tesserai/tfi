from collections import OrderedDict

import inspect
import re
import importlib

import tinydb
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from .arxiv2bib import arxiv2bib

class ArxivBibtexRepo(object):
    def __init__(self, path, resolver=arxiv2bib):
        storage = CachingMiddleware(JSONStorage)
        self._db = tinydb.TinyDB(path, storage=storage)
        self._resolver = resolver
        self._flush = storage.flush

    def close(self):
        self._db.close()

    def resolve(self, arxiv_ids):
        Arxiv = tinydb.Query()
        results = []
        needed = []
        ix = 0
        for arxiv_id in arxiv_ids:
            found = self._db.search(Arxiv.arxiv_id == arxiv_id)
            if len(found) > 0:
                results.append(found[0]['bibtex'])
            else:
                results.append(None)
                needed.append((arxiv_id, ix))
            ix += 1

        if len(needed) > 0:
            resolved = self._resolver([id for id, ix in needed])
            for reference, (id, ix) in zip(resolved, needed):
                bibtex = reference.bibtex()
                self._db.insert(OrderedDict([('arxiv_id', id), ('bibtex', bibtex)]))
                results[ix] = bibtex
            self._flush()

        return results

arxiv_pdf_re = re.compile("(?:https?://)?(?:www\.)?arxiv\.org/pdf/(\d+\.\d+(?:v\d+)?).pdf", re.I)
arxiv_url_re = re.compile("(?:https?://)?(?:www\.)?arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)", re.I)
arxiv_ref_re = re.compile("arxiv: *(\d+\.\d+)", re.I)

def _discover_matches(p, s):
    return [(m.span(), m.groups()[0]) for m in p.finditer(s)]

def discover_arxiv_ids(s):
    ids = []

    ids.extend(_discover_matches(arxiv_url_re, s))
    ids.extend(_discover_matches(arxiv_pdf_re, s))
    ids.extend(_discover_matches(arxiv_ref_re, s))
    return ids

def discover_arxiv_ids_in_doc(fn):
    refs = []
    # refs.extend(discover_arxiv_ids(inspect.getdoc(importlib.import_module(fn.__module__))))
    refs.extend(discover_arxiv_ids(inspect.getdoc(fn)))
    ref_set = set()
    return [ref for ref in refs if not (ref in ref_set or ref_set.add(ref))]

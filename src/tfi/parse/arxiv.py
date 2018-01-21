import inspect
import re

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

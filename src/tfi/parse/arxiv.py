import inspect
import re

def _as_re(subexp):
  return re.compile("\\[ *%s *\\]|%s" % (subexp, subexp), re.I)


arxiv_subexp = "(\\d+\\.\\d+(?:v\\d+)?)"
arxiv_host = "(?:https?://)?(?:www\\.)?arxiv\\.org"
arxiv_pdf_subexp = "%s/pdf/%s.pdf" % (arxiv_host, arxiv_subexp)
arxiv_url_subexp = "%s/abs/%s" % (arxiv_host, arxiv_subexp)
arxiv_pdf_re = _as_re(arxiv_pdf_subexp)
arxiv_url_re = _as_re(arxiv_url_subexp)
arxiv_ref_re = _as_re("arxiv: *(\\d+\\.\\d+)")

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

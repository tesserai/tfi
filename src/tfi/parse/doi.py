import inspect
import re

def _as_re(subexp):
  return re.compile("\\[ *%s *\\]|%s" % (subexp, subexp), re.I)


doi_subexp = "10.\\d{4,9}/[-._;()/:A-Z0-9]+"
doi_url_re = _as_re("(?:https?://)?dx.doi.org/(%s)" % doi_subexp)
doi_ref_re = _as_re("doi: *(%s)" % doi_subexp)

def _discover_matches(p, s):
    return [(m.span(), m.groups()[0]) for m in p.finditer(s)]

def discover_dois(s):
    ids = []

    ids.extend(_discover_matches(doi_url_re, s))
    ids.extend(_discover_matches(doi_ref_re, s))

    print("discover_dois", ids, "in", s)
    return ids

def discover_dois_in_doc(fn):
    refs = []
    # refs.extend(discover_dois(inspect.getdoc(importlib.import_module(fn.__module__))))
    refs.extend(discover_dois(inspect.getdoc(fn)))
    ref_set = set()
    return [ref for ref in refs if not (ref in ref_set or ref_set.add(ref))]


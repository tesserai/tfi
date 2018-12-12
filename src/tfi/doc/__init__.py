import base64
import json
import sys

from collections import OrderedDict as _OrderedDict

from tfi.tensor.frame import TensorFrame as _TensorFrame
from tfi.parse.docstring import GoogleDocstring as _GoogleDocstring
from tfi.parse.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.parse.doi import discover_dois as _discover_dois
from tfi.parse.python import parse_example_args as _parse_example_args
from tfi.resolve.bibtex import BibtexRepo as _BibtexRepo
from tfi.resolve.arxiv2bib import arxiv2bib as _arxiv2bib
from tfi.resolve.doi2bib import doi2bib as _doi2bib
from tfi.resolve.git import git_authorship as _git_authorship
from tfi.resolve.git import GitUserRepo as _GitUserRepo
from tfi.doc.template import render

from tfi.parse.biblib import bib as _bib

from pprint import pprint

def _detect_paragraph_citations(resolve_citation_id, paragraph):
    # Should return index ranges with ids, along with ids
    # themselves. Then we can replace them directly in paragraph with
    # references to their ids and accumulate an id citation.
    matches = [
        *[(span, 'arxiv_id', id) for (span, id) in _discover_arxiv_ids(paragraph)],
        *[(span, 'doi', id) for (span, id) in _discover_dois(paragraph)],
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
            citation_id = resolve_citation_id(id_type, id)
        except Exception as ex:
            print(ex)
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

def record_documentation(model):
    if not hasattr(model, '__tfi_documentation__'):
        model.__tfi_documentation__ = documentation(model)
        model.__tfi_saved_fields__.append('__tfi_documentation__')

def documentation(model):
    # if not isinstance(model, Model):
    #     raise Exception("%s is not an instance of Model" % model)

    if hasattr(model, '__tfi_documentation__'):
        return model.__tfi_documentation__

    bibtex_repo = _BibtexRepo("bibtex.json", {"arxiv_id": _arxiv2bib, "doi": _doi2bib})
    github_user_repo = _GitUserRepo("github-users.json")

    bibparser = _bib.Parser()
    citation_ids = {}

    def _resolve_citation_id(id_type, id):
        if id not in citation_ids:
            bibtex = bibtex_repo.resolve(**{id_type: [id]})[0]
            bibparser.parse(bibtex, log_fp=sys.stderr)
            if bibtex.startswith("@article{"):
                citation_ids[id] = bibtex.split(",", 1)[0][len("@article{"):]
            else:
                return id

        return citation_ids[id]

    overview = None
    if model.__tfi_overview__:
        overview = _detect_paragraph_citations(
            _resolve_citation_id,
            model.__tfi_overview__)

    git_authorship_file = None
    if hasattr(model, '__file__'):
        git_authorship_file = model.__file__
    elif hasattr(model, '__tfi_file__'):
        git_authorship_file = model.__tfi_file__
    elif hasattr(model, '__tfi_module__'):
        git_authorship_file = model.__tfi_module__.__file__

    git = None
    authors = []
    if git_authorship_file:
        git = _git_authorship(github_user_repo, git_authorship_file)
        if git:
            authors = git['authors']

    def prep_python_method(method_name, method_doc):
        example_args = {}
        if method_doc.get('example args', None) is not None:
            example_args = _parse_example_args(method_doc['example args'], {'m': model})
        example_result = {}
        try:
            example_result = getattr(model, method_name)(**example_args)
            if isinstance(example_result, _TensorFrame):
                example_result = example_result.dict()
        except Exception as ex:
            print(ex)

        method_text_sections = [v for t, v in method_doc['sections'] if t == 'text']
        method_overview = _detect_paragraph_citations(
            _resolve_citation_id,
            "\n".join([l for t in method_text_sections for l in t]))

        return {
            "name": method_name,
            "overview": method_overview,
            "sections": method_doc['sections'],
            "args": method_doc['args'],
            "returns": method_doc['returns'],
            # "example usage": [example_python_src],
            # "example usage": [example_curl_src],
            "example args": example_args,
            "example result": example_result,
        }

    def prep_hyperparameter(name, type, value, doc):
        kind = ["value", "was", repr(value)]
        return (name, " ".join(kind), doc)

    return {
        "title": model.__tfi_name__ if hasattr(model, '__tfi_name__') else type(model).__name__,
        "source": {
            "url": git["url"],
            "label": git["label"],
            "commit": git["commit"][:7],
        } if git else {},
        "authors": [
           *[
                {
                    "name": author['name'],
                    "url": author['url'],
                    "role_noun": "Commits",
                    "role_url": author['commits_url'],
                }
                for author in authors
            ],
        ],
        "overview": overview,
        "methods": [
            prep_python_method(method_name, method_doc)
            for method_name, method_doc in model.__tfi_signature_defs_docs__.items()
        ],
        "facets_overview_proto": model.__tfi_facets_overview_proto__ if hasattr(model, '__tfi_facets_overview_proto__') else None,
        "hyperparameters": [
            prep_hyperparameter(*hparam_tuple)
            for hparam_tuple in model.__tfi_hyperparameters__
        ] if hasattr(model, '__tfi_hyperparameters__') else [],
        "implementation_notes": [],
        "references": _OrderedDict([
            (k, _OrderedDict(v))
            for k, v in reversed(bibparser.get_entries().items())
        ]),
    }

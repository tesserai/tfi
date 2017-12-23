import base64
import sys

from collections import OrderedDict as _OrderedDict

from tfi.parse.docstring import GoogleDocstring as _GoogleDocstring
from tfi.resolve.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.resolve.arxiv import ArxivBibtexRepo as _ArxivBibtexRepo
from tfi.resolve.arxiv2bib import arxiv2bib as _arxiv2bib
from tfi.resolve.git import git_authorship as _git_authorship
from tfi.resolve.git import GitUserRepo as _GitUserRepo
from tfi.doc import template as _template

from tfi.parse.biblib import bib as _bib

from pprint import pprint

def _detect_paragraph_citations(resolve_citation_id, paragraph):
    # Should return index ranges with arxiv_ids, along with arxiv_ids
    # themselves. Then we can replace them directly in paragraph with
    # references to their arxiv_ids and accumulate an arxiv_id citation.
    arxiv_id_matches = _discover_arxiv_ids(paragraph)

    # Replace span in reverse with archive_id
    arxiv_id_matches.sort(reverse=True)
    print("arxiv_id_matches", arxiv_id_matches)

    new_paragraph_parts = []
    prev_start = len(paragraph)

    # Rewrite paragraph in reverse, so span indexes are correct.
    # Replace URLs to papers with proper references to papers.
    # Accumulate paragraph pieces and then join them at the very
    # end to avoid wasting string allocations.
    for (start, end), arxiv_id in arxiv_id_matches:
        citation_id = resolve_citation_id(arxiv_id)
        suffix = paragraph[end+1:prev_start]
        print("suffix", suffix, (start, end), end+1, prev_start-end)
        new_paragraph_parts.append(suffix)
        new_paragraph_parts.append("]_ ")
        new_paragraph_parts.append(citation_id)
        new_paragraph_parts.append(" [")
        prev_start = start
    if prev_start != 0:
        new_paragraph_parts.append(paragraph[0:prev_start])

    new_paragraph_parts.reverse()
    return "".join(new_paragraph_parts)

def documentation(model):
    # if not isinstance(model, Base):
    #     raise Exception("%s is not an instance of Base" % model)

    arxiv_repo = _ArxivBibtexRepo("arxiv.json", _arxiv2bib)
    github_user_repo = _GitUserRepo("github-users.json")

    bibparser = _bib.Parser()
    citation_ids = {}

    def _resolve_citation_id(arxiv_id):
        if arxiv_id in citation_ids:
            return citation_ids[arxiv_id]
        else:
            bibtex = arxiv_repo.resolve([arxiv_id])[0]
            bibparser.parse(bibtex, log_fp=sys.stderr)
            if bibtex.startswith("@article{"):
                return bibtex.split(",", 1)[0][len("@article{"):]
            return arxiv_id

    overview = None
    model_doc_sections = []
    if model.__doc__:
        model_doc = _GoogleDocstring(obj=model).result()
        model_doc_sections = model_doc['sections']

        # TODO(adamb) Should we be parsing rst here? Probably not. We probably
        # want to just emit valid (possibly rewritten) rst. Then we'll parse
        # rst for real when rendering HTML.
        text_sections = [v for t, v in model_doc_sections if t == 'text']
        overview = _detect_paragraph_citations(
            _resolve_citation_id,
            "\n".join([l for t in text_sections for l in t]))

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

    def prep_method(method_name, method_doc):
        example_args_list = [
            (name, "\n".join(doc))
            for name, type, doc in method_doc['example args']
        ]

        # TODO(adamb) Confirm we can properly parse k as an id and v alone.
        example_args_kw_src = ", ".join([
            "%s=%s" % (k, v)
            for k, v in example_args_list
        ])

        example_args_src = """
import tfi.data
_ = dict(%s)
""" % example_args_kw_src
        g = {}
        l = {'m': model}
        exec(example_args_src, g, l)
        example_args = l['_']

        example_usage_src = "m.%s(%s)" % (method_name, example_args_kw_src)
        example_result = {}
        try:
            example_result = getattr(model, method_name)(**example_args)
        except Exception as ex:
            print(ex)

        d = {
            "name": method_name,
            "sections": method_doc['sections'],
            "args": method_doc['args'],
            "example usage": [example_usage_src],
            "example args": example_args,
            "example result": example_result,
            "returns": method_doc['returns'],
        }
        return d

    def prep_hyperparameter(name, type, value, doc):
        kind = ["value", "was", repr(value)]
        return (name, " ".join(kind), doc)

    return {
        "title": model.__name__ if hasattr(model, '__name__') else type(model).__name__,
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
                    "affiliation_name": "Code Contributor",
                    "affiliation_url": author['commits_url'],
                }
                for author in authors
            ],
        ],
        "overview": overview,
        "methods": [
            prep_method(method_name, method_doc)
            for method_name, method_doc in model.__tfi_signature_def_docs__.items()
        ],
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

def record_documentation(model):
    model.__tfi_documentation__ = documentation(model)

def render(model):
    if hasattr(model, '__tfi_documentation__'):
        template_args = model.__tfi_documentation__
    else:
        template_args = documentation(model)
    return _template.render(**template_args)

def save(path, model):
    template_args = documentation(model)
    _template.write(path, **template_args)

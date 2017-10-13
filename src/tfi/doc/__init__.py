from tfi.doc.docstring import GoogleDocstring as _GoogleDocstring
from tfi.doc.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.doc.arxiv import ArxivBibtexRepo as _ArxivBibtexRepo
from tfi.doc.arxiv2bib import arxiv2bib as _arxiv2bib
from tfi.doc.git import git_authorship as _git_authorship
from tfi.doc import template as _template

def save(path, model):
    # if not isinstance(model, Base):
    #     raise Exception("%s is not an instance of Base" % model)

    arxiv_repo = _ArxivBibtexRepo("arxiv.json", _arxiv2bib)
    references = []
    if model.__doc__:
        model_doc_sections, _ = _GoogleDocstring(obj=model).result()
        references.extend(arxiv_repo.resolve(_discover_arxiv_ids(model)))
    else:
        model_doc_sections = []

    if hasattr(model, '__file__'):
        git = _git_authorship(model.__file__)
    else:
        git = {'authors': []}

    if len(model_doc_sections) > 0:
        subhead = "\n".join(model_doc_sections[0][1])
        model_doc_sections = model_doc_sections[1:]
    else:
        subhead = ""

    template_args = {
        "title": model.__name__ if hasattr(model, '__name__') else type(model).__name__,
        "subhead": subhead,
        "authors": [
           *[
                {
                    "name": author['name'],
                    "url": "mailto:%s" % author['email'],
                    "affiliation_name": "Code Contributor",
                    "affiliation_url": author['commits_url'],
                }
                for author in git['authors']
            ],
        ],
        "sections": model_doc_sections,
        "methods": [
            {
                "name": method_name,
                "sections": method_doc['sections'],
                "args": method_doc['args'],
                "returns": method_doc['returns'],
            }
            for method_name, method_doc in model.__tfi_signature_def_docs__.items()
        ],
        "references": references,
    }
    _template.write(path, **template_args)

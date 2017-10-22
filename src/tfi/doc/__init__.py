from tfi.doc.docstring import GoogleDocstring as _GoogleDocstring
from tfi.doc.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.doc.arxiv import ArxivBibtexRepo as _ArxivBibtexRepo
from tfi.doc.arxiv2bib import arxiv2bib as _arxiv2bib
from tfi.doc.git import git_authorship as _git_authorship
from tfi.doc.git import GitUserRepo as _GitUserRepo
from tfi.doc import template as _template

def save(path, model):
    # if not isinstance(model, Base):
    #     raise Exception("%s is not an instance of Base" % model)

    arxiv_repo = _ArxivBibtexRepo("arxiv.json", _arxiv2bib)
    github_user_repo = _GitUserRepo("github-users.json")

    references = []
    if model.__doc__:
        model_doc_sections, _ = _GoogleDocstring(obj=model).result()
        references.extend(arxiv_repo.resolve(_discover_arxiv_ids(model)))
    else:
        model_doc_sections = []

    git_authorship_file = None
    if hasattr(model, '__file__'):
        git_authorship_file = model.__file__
    elif hasattr(model, '__tfi_file__'):
        git_authorship_file = model.__tfi_file__
    elif hasattr(model, '__tfi_module__'):
        git_authorship_file = model.__tfi_module__.__file__

    git = _git_authorship(github_user_repo, git_authorship_file)

    if len(model_doc_sections) > 0:
        subhead = "\n".join(model_doc_sections[0][1])
        model_doc_sections = model_doc_sections[1:]
    else:
        subhead = ""

    def shorten_author_name(name, max):
        if len(name) < max:
            return name

        parts = name.split(" ")
        # Shorten to first initial of each part
        parts[:-1] = [
            "%s." % part[0] if len(part) > 2 and part[0].isalpha() else part
            for part in parts[:-1]
        ]
        return " ".join(parts)

    template_args = {
        "title": model.__name__ if hasattr(model, '__name__') else type(model).__name__,
        "subhead": subhead,
        "source": {
            "url": git["url"],
            "label": git["label"],
            "commit": git["commit"][:7],
        },
        "authors": [
           *[
                {
                    "name": shorten_author_name(author['name'], 14),
                    "url": author['url'],
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

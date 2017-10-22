import base64

from tfi.as_tensor import maybe_as_tensor
from tfi.data import terminal
from tfi.data import _encode

from tfi.doc.docstring import GoogleDocstring as _GoogleDocstring
from tfi.doc.arxiv import discover_arxiv_ids as _discover_arxiv_ids
from tfi.doc.arxiv import ArxivBibtexRepo as _ArxivBibtexRepo
from tfi.doc.arxiv2bib import arxiv2bib as _arxiv2bib
from tfi.doc.git import git_authorship as _git_authorship
from tfi.doc.git import GitUserRepo as _GitUserRepo
from tfi.doc import template as _template

def _run_example(m, example_src):
    s = """
import tfi, numpy as np

m = tfi.saved_model("//tensorflow/magenta/image_stylization")

_ = m.stylize(images=[tfi.data.file("data/kodim01.png")],
          style_weights=np.identity(m.hparams().num_styles)[7])[0][0]
    """

    e = """
_prev_file = tfi.data.file
__trapped__ = []
def wrap_file(p):
    global _prev_file, __trapped__
    r = _prev_file(p)
    __trapped__.append((p, r))
    return r

tfi.data.file = wrap_file
    """

    import re
    example_src = s
    source = re.sub(r'm = tfi.saved_model.+', e, example_src)

    l = {'m': m}
    g = {}
    exec(source, g, l)

    return g['__trapped__'], l['_']

def save(path, model):
    # if not isinstance(model, Base):
    #     raise Exception("%s is not an instance of Base" % model)

    arxiv_repo = _ArxivBibtexRepo("arxiv.json", _arxiv2bib)
    github_user_repo = _GitUserRepo("github-users.json")

    references = []
    model_doc_sections = []
    if model.__doc__:
        model_doc = _GoogleDocstring(obj=model).result()
        model_doc_sections = model_doc['sections']
        references.extend(arxiv_repo.resolve(_discover_arxiv_ids(model)))

    git_authorship_file = None
    if hasattr(model, '__file__'):
        git_authorship_file = model.__file__
    elif hasattr(model, '__tfi_file__'):
        git_authorship_file = model.__tfi_file__
    elif hasattr(model, '__tfi_module__'):
        git_authorship_file = model.__tfi_module__.__file__

    print("git_authorship_file", git_authorship_file)
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

    class escaped(object):
        def __init__(self, s):
            self.s = s
        def __repr__(self):
            return self.s

    def img_png_html(d):
        return escaped("""<img src="data:image/png;base64,%s">""" % base64.b64encode(d).decode())

    def text_plain(d):
        print("text_plain", d)
        return d

    def inspect(o):
        tensor = maybe_as_tensor(o, None, None)
        accept_mimetypes = {"image/png": img_png_html, "text/plain": text_plain}
        val = _encode(tensor, accept_mimetypes)
        if val is None:
            val = o
        return '%r' % val

    def prep_method(method_name, method_doc):
        example_args = None
        for t, r in method_doc['sections']:
            if t == 'example args':
                example_args = r
                break

        example_expanded = []
        example_result = None
        if method_name == "stylize" and example_args is not None:
            trapped, result = _run_example(model, example_args)

            for name, t in trapped:
                example_expanded.append((name, inspect(t)))

            example_result = inspect(result)

        return {
            "name": method_name,
            "sections": method_doc['sections'],
            "args": method_doc['args'],
            "example args": example_args,
            "example expanded": example_expanded,
            "example result": example_result,
            "returns": method_doc['returns'],
        }

    print("references", references)

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
            prep_method(method_name, method_doc)
            for method_name, method_doc in model.__tfi_signature_def_docs__.items()
        ],
        "references": references,
    }
    _template.write(path, **template_args)

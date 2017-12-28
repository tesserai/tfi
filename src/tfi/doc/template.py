from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, BashLexer

from jinja2 import Template as JinjaTemplate

from tfi.format.html.bibtex import citation_bibliography_html
from tfi.format.html.python import inspect_html
from tfi.format.html.rst import parse_rst as _parse_rst

from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateGoogleStyle

from collections import OrderedDict

import json
import shlex

_page_template_path = __file__[:-2] + "html"
if _page_template_path.endswith("__init__.html"):
    _page_template_path = _page_template_path.replace("__init__.html", __name__.split('.')[-1] + '.html')

def _read_style_fragment():
    hf = HtmlFormatter(style='paraiso-dark')
    _style_fragment = hf.get_style_defs('.language-source')

    # _style_file = __file__[:-2] + "css"
    # if _style_file.endswith("__init__.css"):
    #     _style_file = _style_file.replace("__init__.css", __name__.split('.')[-1] + '.css')
    # with open(_style_file, encoding="utf-8") as f:
    #     _style_fragment += f.read()
    return _style_fragment

# ['default', 'emacs', 'friendly', 'colorful', 'autumn', 'murphy', 'manni', 'monokai', 'perldoc', 'pastie', 'borland', 'trac', 'native', 'fruity', 'bw',
# 'vim', 'vs', 'tango', 'rrt', 'xcode', 'igor', 'paraiso-light', 'paraiso-dark', 'lovelace', 'algol', 'algol_nu', 'arduino', 'rainbow_dash', 'abap']

def render(
        source,
        authors,
        title,
        overview,
        methods,
        hyperparameters,
        implementation_notes,
        references,
        host,
        extra_scripts=""):

    with open(_page_template_path, encoding='utf-8') as f:
        t = JinjaTemplate(f.read())

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

    def python_source_for(method):
        return "m.%s(%s)" % (method['name'], ", ".join(
            [
                "%s=%r" % (k, v)
                for k, v in method['example args'].items()
            ]
        ))
    
    def curl_source_for(method):
        def json_default(o):
            print("json_default", o, type(o), dir(o))
            if hasattr(o, '__json__'):
                return o.__json__()
            raise TypeError("Unserializable object {} of type {}".format(o, type(o)))

        def quoted_json_dumps(v):
            try:
                print("v", type(v), dir(v))
                s = json.dumps(v, default=json_default)
                return shlex.quote(s)
            except ValueError as ex:
                print("v", v)
                print(ex)
                return None

        return "curl %s" % " \\\n   ".join([
            "http://%s/%s" % (host, method['name']),
            *[
                "-F %s=%s" % (k, quoted_json_dumps(v))
                for k, v in method['example args'].items()
            ],
        ])

    def highlight_shell_source(s):
        s2 = highlight(s,
                  BashLexer(),
                  HtmlFormatter(cssclass='language-source language-curl'))
        s2 = s2.replace("</pre>", """<span class="line-numbers">%s</span></pre>""" % ("".join(["<span></span>"] * (1 + s.count("\n")))))
        return s2

    def highlight_python_source(s, more_style_config={}):
        style_config = CreateGoogleStyle()
        style_config.update(more_style_config)
        s, _ = FormatCode(s, style_config=style_config)
        s = s.strip()

        s2 = highlight(s,
                  PythonLexer(),
                  HtmlFormatter(cssclass='language-source language-python'))
        s2 = s2.replace("</pre>", """<span class="line-numbers">%s</span></pre>""" % ("".join(["<span></span>"] * (1 + s.count("\n")))))
        return s2

    def highlight_python_value(o, max_width=None, max_seq_length=None):
        s, xform = inspect_html(o, max_width=max_width, max_seq_length=max_seq_length)
        r = highlight(s,
                  PythonLexer(),
                  HtmlFormatter(cssclass='language-python'))
        return xform(r)

    citation_id = 0
    citation_label_by_refname = {}
    def citation_label_by_refname_fn(citation_refname):
        nonlocal citation_id
        if citation_refname not in citation_label_by_refname:
            citation_id += 1
            citation_label_by_refname[citation_refname] = str(citation_id)
        return citation_label_by_refname[citation_refname]

    def reference_fn(citation_refname):
        return references[citation_refname]

    # TODO(adamb) What about arxiv ids already within []_ ??
    parsed = _parse_rst(overview, "<string>",
            citation_label_by_refname_fn, reference_fn)

    hover_divs = []
    hover_divs.extend(parsed['hover_divs'])
    def refine_method(method):
        method = dict(method)
        parsed_overview = _parse_rst(method['overview'], "<string>",
                citation_label_by_refname_fn, reference_fn)
        method['overview'] = parsed_overview['body']
        hover_divs.extend(parsed_overview['hover_divs'])
        return method

    methods = [
        refine_method(method)
        for method in methods
    ]

    return t.render(
            python_source_for=python_source_for,
            curl_source_for=curl_source_for,
            highlight_shell_source=highlight_shell_source,
            highlight_python_value=highlight_python_value,
            highlight_python_source=highlight_python_source,
            extra_scripts=extra_scripts,
            shorten_author_name=shorten_author_name,
            style_fragment=_read_style_fragment(),
            title=parsed['title'],
            subhead=parsed['subtitle'],
            bibliography_cite=citation_bibliography_html,
            overview=parsed['body'],
            authors=authors,
            hyperparameters=hyperparameters,
            implementation_notes=implementation_notes,
            hover_divs=hover_divs,
            references=OrderedDict([
                (refname, references[refname])
                for refname in citation_label_by_refname
            ]),
            host=host,
            methods=methods)

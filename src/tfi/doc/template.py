from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, BashLexer

from jinja2 import Template as JinjaTemplate

from tfi.format.html.bibtex import citation_bibliography_html
from tfi.format.html.python import inspect_html
from tfi.format.html.rst import parse_rst as _parse_rst

from yapf.yapflib.yapf_api import FormatCode
from yapf.yapflib.style import CreateGoogleStyle

import json
import shlex

_page_template_path = __file__[:-2] + "html"
if _page_template_path.endswith("__init__.html"):
    _page_template_path = _page_template_path.replace("__init__.html", __name__.split('.')[-1] + '.html')

def _read_style_fragment():
    _style_fragment = HtmlFormatter(style='paraiso-dark').get_style_defs('.language-python')

    _style_file = __file__[:-2] + "css"
    if _style_file.endswith("__init__.css"):
        _style_file = _style_file.replace("__init__.css", __name__.split('.')[-1] + '.css')
    with open(_style_file, encoding="utf-8") as f:
        _style_fragment += f.read()
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
        host="localhost:5000",
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
        def quoted_json_dumps(v):
            s = json.dumps(v, default=lambda o: o.__json__() if hasattr(o, '__json__') else o)
            return shlex.quote(s)

        return "curl %s" % " \\\n     ".join([
            "--request POST",
            "--url http://%s/%s" % (host, method['name']),
            *[
                "--form %s=%s" % (k, quoted_json_dumps(v))
                for k, v in method['example args'].items()
            ],
        ])

    def highlight_shell_source(s):
        return highlight(s,
                  BashLexer(),
                  HtmlFormatter(cssclass='language-python'))

    def highlight_python_source(s, more_style_config={}):
        return highlight_shell_source(s)

        style_config = CreateGoogleStyle()
        style_config.update(more_style_config)
        s, _ = FormatCode(s, style_config=style_config)

        return highlight(s,
                  PythonLexer(),
                  HtmlFormatter(cssclass='language-python'))

    def highlight_python_value(o, max_width=None, max_seq_length=None):
        s, xform = inspect_html(o, max_width=max_width, max_seq_length=max_seq_length)
        r = highlight(s,
                  PythonLexer(),
                  HtmlFormatter(cssclass='language-python'))
        return xform(r)

    # TODO(adamb) What about arxiv ids already within []_ ??
    parsed = _parse_rst(overview, "<string>", references)

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
            hover_divs=parsed['hover_divs'],
            references=references,
            host=host,
            methods=methods)
